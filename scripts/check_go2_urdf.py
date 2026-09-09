"""Compare converted Go2 URDF physics against the original composed USD.

Run with the Isaac Lab Python environment. The reference USD must include its
Props/instanceable_meshes.usd dependency (or use a resolvable asset URL).
"""

import argparse
import tempfile
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.converters import UrdfConverter
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from mglf_lab.assets.go2 import UNITREE_GO2_URDF_CFG


def snapshot(stage):
    """Normalize USD hierarchy differences into comparable physical properties."""
    bodies, joints = {}, {}
    colliders = defaultdict(list)
    for prim in stage.Traverse():
        def value(name):
            return prim.GetAttribute(name).Get()

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rotation = np.array(Gf.Matrix3d(Gf.Quatd(value("physics:principalAxes"))))
            bodies[prim.GetName()] = {
                "mass": value("physics:mass"),
                "com": value("physics:centerOfMass"),
                "inertia": rotation.T @ np.diag(value("physics:diagonalInertia")) @ rotation,
                "pose": np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)),
            }
        if prim.IsA(UsdPhysics.Joint):
            row = {name: value(name) for name in ("physics:localPos0", "physics:localPos1")}
            row["type"] = prim.GetTypeName()
            for name in ("physics:body0", "physics:body1"):
                row[name] = prim.GetRelationship(name).GetTargets()[0].name
            if prim.IsA(UsdPhysics.RevoluteJoint):
                for name in (
                    "physics:lowerLimit", "physics:upperLimit",
                    "drive:angular:physics:maxForce", "physxJoint:maxJointVelocity",
                ):
                    row[name] = value(name)
                axis = {
                    "X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1),
                }[value("physics:axis")]
                for name in ("physics:localRot0", "physics:localRot1"):
                    row[name] = Gf.Rotation(Gf.Quatd(value(name))).TransformDir(axis)
            else:
                for name in ("physics:localRot0", "physics:localRot1"):
                    row[name] = np.array(Gf.Matrix3d(Gf.Quatd(value(name))))
            joints[prim.GetName()] = row
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetParent()
        while not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
            assert body, f"Collider has no rigid body: {prim.GetPath()}"
        row = {
            "type": prim.GetTypeName(),
            "pose": np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)),
        }
        for name in ("size", "radius", "height", "axis"):
            attribute = prim.GetAttribute(name)
            if attribute:
                row[name] = attribute.Get()
        colliders[body.GetName()].append(row)
    return {"bodies": bodies, "joints": joints, "colliders": dict(colliders)}


def compare(reference, imported, path=""):
    """Fail with the exact property path when physical values differ."""
    if isinstance(reference, dict):
        assert reference.keys() == imported.keys(), (path, reference.keys(), imported.keys())
        for key in reference:
            compare(reference[key], imported[key], path + "/" + key)
    elif isinstance(reference, list):
        assert len(reference) == len(imported), (path, len(reference), len(imported))
        for index, (left, right) in enumerate(zip(reference, imported)):
            compare(left, right, path + "/" + str(index))
    elif isinstance(reference, str):
        assert reference == imported, (path, reference, imported)
    else:
        np.testing.assert_allclose(reference, imported, rtol=2e-6, atol=2e-7, err_msg=path)


try:
    with tempfile.TemporaryDirectory(prefix="go2_parity_") as directory:
        cfg = UNITREE_GO2_URDF_CFG.spawn.copy()
        cfg.usd_dir = directory
        cfg.force_usd_conversion = True
        converted = UrdfConverter(cfg)
        stages = [Usd.Stage.Open(path) for path in (args.reference_usd, converted.usd_path)]
        reference, imported = map(snapshot, stages)
        # These counts also detect missing collision reference layers.
        assert len(reference["bodies"]) == 19
        assert len(reference["joints"]) == 18
        assert sum(map(len, reference["colliders"].values())) == 27
        compare(reference, imported)
        print("PASS: 19 bodies, 18 joints and 27 collision shapes match.", flush=True)

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu"))
        original_cfg = UNITREE_GO2_CFG.copy()
        original_cfg.prim_path = "/World/Original"
        original_cfg.spawn.usd_path = args.reference_usd
        imported_cfg = UNITREE_GO2_URDF_CFG.copy()
        imported_cfg.prim_path = "/World/Imported"
        imported_cfg.spawn.usd_dir = directory
        imported_cfg.init_state.pos = (2.0, 0.0, 0.4)
        original, imported = [Articulation(c) for c in (original_cfg, imported_cfg)]
        sim.reset()
        for robot in (original, imported):
            robot.update(sim.get_physics_dt())
        assert original.joint_names == imported.joint_names
        assert original.body_names == imported.body_names
        for name in (
            "default_joint_pos", "default_joint_vel", "joint_pos_limits",
            "soft_joint_pos_limits", "joint_vel_limits", "joint_effort_limits",
            "joint_stiffness", "joint_damping", "joint_armature",
            "joint_friction_coeff", "default_mass", "default_inertia",
        ):
            torch.testing.assert_close(
                getattr(original.data, name), getattr(imported.data, name),
                rtol=2e-5, atol=2e-6, msg=name,
            )
        for name in (
            "get_masses", "get_coms", "get_inertias", "get_dof_limits",
            "get_dof_max_velocities", "get_dof_max_forces", "get_dof_stiffnesses",
            "get_dof_dampings", "get_dof_armatures", "get_dof_friction_coefficients",
        ):
            torch.testing.assert_close(
                getattr(original.root_physx_view, name)(),
                getattr(imported.root_physx_view, name)(),
                rtol=2e-5, atol=2e-6, msg=name,
            )
        print("PASS: PhysX body/joint order, limits, inertials and actuator parameters match.", flush=True)
        sim.stop()
        sim.clear()
finally:
    app.close(wait_for_replicator=False)
