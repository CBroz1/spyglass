"""Importing pretrained models whose training videos are not in Spyglass.

Plan: `.claude/TASKS_ZOO.md`. Two cases drive this — pilot/scoping videos that
should not be registered as sessions, and DLC Model Zoo models whose training
videos do not exist at all.

Every test here is RED until the matching task lands. Task IDs in the class
docstrings map to TASKS_ZOO.md §4.

Fixture rules (from the #1680 post-mortem, `CLAUDE.md`): no session-scoped
fixture may insert `Session`/`Nwbfile`, and teardown uses
``super_delete(warn=False, safemode=False)`` — plain ``delete()`` triggers
``cautious_delete``'s *global* external-file sweep and deletes other suites'
analysis files.
"""

import pathlib

import pytest

from tests.conftest import skip_if_no_pose

# DLC ships four SuperAnimal project configs. Names are 16-24 chars, which is
# why they live in `model_name` and never in the 32-char `model_id` (a prefix
# that long truncates the identifying hash -- see TASKS_ZOO.md §3.6).
ZOO_MODELS = [
    "superanimal_topviewmouse",
    "superanimal_quadruped",
    "superanimal_bird",
    "superanimal_humanbody",
]


@pytest.fixture
def requires_network():
    """Skip unless the SuperAnimal weights are already cached on disk.

    Deliberately checks the cache rather than connectivity. One model is
    ~646 MB (152 MB pose + 494 MB detector for topviewmouse/hrnet_w32), so a
    test run must never fetch it implicitly — CI also passes ``--no-pose``, so
    these never run there regardless.

    Pre-fetch once with::

        python -m spyglass.position.v2.utils.fetch_dlc_zoo superanimal_topviewmouse

    Returns
    -------
    callable
        ``_require(dataset, model_name=None)`` -- skips if absent. ``None``
        means "whichever backbone the dataset offers first", since backbones
        differ per dataset.
    """

    def _require(dataset, model_name=None):
        from spyglass.position.v2.utils import fetch_dlc_zoo

        if not fetch_dlc_zoo.zoo_catalog():
            pytest.skip("deeplabcut 3.x / dlclibrary not installed")
        if not fetch_dlc_zoo.is_cached(dataset, model_name):
            pytest.skip(
                f"zoo weights not cached for {dataset}. Pre-fetch with "
                "`python -m spyglass.position.v2.utils.fetch_dlc_zoo "
                f"{dataset}`"
            )
        return fetch_dlc_zoo.snapshot_path(dataset, model_name)

    return _require


class TestZooCatalog:
    """`fetch_dlc_zoo` — catalog lookup, the basis of `Model.load` dispatch.

    No weights needed: the catalog ships with dlclibrary, and snapshot paths
    are deterministic whether or not the file exists.
    """

    @pytest.fixture
    def zoo(self):
        from spyglass.position.v2.utils import fetch_dlc_zoo

        if not fetch_dlc_zoo.zoo_catalog():
            pytest.skip("deeplabcut 3.x / dlclibrary not installed")
        return fetch_dlc_zoo

    def test_catalog_lists_superanimals(self, zoo):
        catalog = zoo.zoo_catalog()
        assert set(ZOO_MODELS) <= set(catalog)
        assert all(backbones for backbones in catalog.values())

    def test_is_zoo_model(self, zoo):
        assert zoo.is_zoo_model("superanimal_topviewmouse")
        assert not zoo.is_zoo_model("/some/path/config.yaml")
        assert not zoo.is_zoo_model("superanimal_notathing")

    def test_default_backbone_is_dataset_specific(self, zoo):
        """No single default works: bird has only resnet_50, humanbody only
        rtmpose_x. A hardcoded `hrnet_w32` is wrong for half the catalog."""
        catalog = zoo.zoo_catalog()
        for dataset, backbones in catalog.items():
            assert zoo.default_backbone(dataset) in backbones

    def test_resolve_backbone_rejects_unavailable(self, zoo):
        catalog = zoo.zoo_catalog()
        dataset = "superanimal_bird"
        if "hrnet_w32" in catalog.get(dataset, []):
            pytest.skip("catalog changed; bird now offers hrnet_w32")
        with pytest.raises(ValueError, match="offers"):
            zoo.resolve_backbone(dataset, "hrnet_w32")

    def test_unknown_dataset_lists_the_catalog(self, zoo):
        with pytest.raises(ValueError, match="superanimal"):
            zoo.default_backbone("superanimal_notathing")

    def test_snapshot_path_does_not_download(self, zoo):
        """Deterministic path, so callers can check before committing ~646MB."""
        path = zoo.snapshot_path("superanimal_topviewmouse")
        assert path.name.startswith("superanimal_topviewmouse")
        assert path.suffix == ".pt"
        assert zoo.is_cached("superanimal_topviewmouse") == path.exists()

    def test_is_cached_is_false_for_unknown(self, zoo):
        """Unknown names must answer False, not raise -- `load` dispatch
        calls this on arbitrary strings."""
        assert zoo.is_cached("superanimal_notathing") is False


class TestExternalVideoTable:
    """Z01 — `VidFileGroup.ExternalVideo` records unregistered training media.

    The point is honesty: the videos are documented and queryable without
    claiming they are recording sessions. A fabricated `Session` would be
    indistinguishable from real data downstream.
    """

    def test_part_table_exists(self, VidFileGroup):
        assert hasattr(VidFileGroup, "ExternalVideo")

    def test_path_is_nullable(self, VidFileGroup):
        """Zoo configs ship `video_sets:` empty — source without paths."""
        heading = VidFileGroup.ExternalVideo().heading
        assert "path" in heading.names
        assert heading.attributes["path"].nullable

    def test_records_source_without_path(self, VidFileGroup, make_vid_group):
        """A zoo import has provenance but no file paths."""
        gid = make_vid_group("zoo_extvid_probe", [])
        VidFileGroup.ExternalVideo().insert1(
            {
                "vid_group_id": gid,
                "external_video_id": 0,
                "path": None,
                "source": "dlc-modelzoo:superanimal_topviewmouse",
                "note": "training videos not publicly available",
            }
        )
        row = (VidFileGroup.ExternalVideo & {"vid_group_id": gid}).fetch1()
        assert row["path"] is None
        assert row["source"].endswith("superanimal_topviewmouse")

    def test_coexists_with_fileless_group(self, VidFileGroup, make_vid_group):
        """External rows are not `File` rows — the group stays session-less.

        This is what keeps the inference guard meaningful: `get_nwb_file()`
        must still refuse this group, so it can never back a `PoseV2` entry.
        """
        gid = make_vid_group("zoo_fileless_probe", [])
        VidFileGroup.ExternalVideo().insert1(
            {
                "vid_group_id": gid,
                "external_video_id": 0,
                "path": "/elsewhere/pilot_video.mp4",
                "source": "external",
                "note": "",
            }
        )
        assert len(VidFileGroup.File & {"vid_group_id": gid}) == 0
        with pytest.raises(ValueError, match="no video files"):
            VidFileGroup().get_nwb_file(gid)


class TestModelName:
    """Z02 — `Model` gains a human-readable name.

    There is currently no such field: `model_id` is a date+hash, so `Model()`
    previews are unreadable. For a zoo model the name *is* the identity.
    """

    def test_attribute_exists(self, pv2_train):
        assert "model_name" in pv2_train.Model().heading.names

    def test_nullable_for_existing_rows(self, pv2_train):
        """Back-compat: pre-existing rows must survive the migration."""
        assert pv2_train.Model().heading.attributes["model_name"].nullable

    def test_populated_from_task_on_import(self, stub_model, pv2_train):
        """Project imports default the name to the DLC `Task`."""
        name = (pv2_train.Model & {"model_id": stub_model["model_id"]}).fetch1(
            "model_name"
        )
        assert name


class TestTrainingMode:
    """Z04 — declared continuation capability.

    Three states, not a boolean: DLC's zoo fine-tune entry point is
    `build_weight_init(super_animal=...)`, which *seeds a new project* rather
    than resuming the imported model.
    """

    def test_attribute_exists(self, pv2_train):
        assert "training_mode" in pv2_train.Model().heading.names

    def test_enum_values(self, pv2_train):
        attr = pv2_train.Model().heading.attributes["training_mode"]
        for value in ("resumable", "weights_only", "inference_only"):
            assert value in attr.type

    def test_locally_trained_model_is_resumable(self, stub_model, pv2_train):
        mode = (pv2_train.Model & {"model_id": stub_model["model_id"]}).fetch1(
            "training_mode"
        )
        assert mode == "resumable"


class TestLoadExternalVideos:
    """Z05 — the pilot-video case: import a real project, skip registration."""

    @pytest.fixture
    def unregistered_project(self, tmp_path_factory):
        """A DLC project whose video cannot match any `VideoFile` row.

        The shared example project's video basename *is* registered by
        `dlc_bootstrapped_session`, and `create_from_dlc_config` falls back to
        basename matching -- so reusing it would resolve and silently test the
        wrong branch. Give the video a unique name instead.
        """
        import yaml

        from tests.position.v2.make_example_dlc_project import make_dlc_project

        out = tmp_path_factory.mktemp("unreg_project", numbered=True)
        config_path = make_dlc_project(out)

        cfg = yaml.safe_load(config_path.read_text())
        old_video = next(iter(cfg["video_sets"]))
        unique = (
            pathlib.Path(old_video).parent / "pilot_scoping_unregistered.avi"
        )
        pathlib.Path(old_video).rename(unique)
        cfg["video_sets"] = {str(unique): {"crop": "0, 640, 0, 480"}}
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return config_path

    def test_load_rejects_unregistered_by_default(
        self, pv2_train, unregistered_project, skip_if_no_dlc
    ):
        """Default stays strict — opting out must be explicit."""
        with pytest.raises(ValueError, match="could not be unambiguously"):
            pv2_train.Model().load(unregistered_project, external_videos=False)

    def test_external_videos_records_config_paths(
        self, pv2_train, unregistered_project, skip_if_no_dlc
    ):
        """`video_sets` entries land in ExternalVideo, not VideoFile."""
        from spyglass.position.v2.video import VidFileGroup

        key = pv2_train.Model().load(unregistered_project, external_videos=True)
        gid = (pv2_train.Model & key).fetch1("vid_group_id")

        assert len(VidFileGroup.File & {"vid_group_id": gid}) == 0
        ext = VidFileGroup.ExternalVideo & {"vid_group_id": gid}
        assert len(ext) == 1
        assert "pilot_scoping_unregistered" in ext.fetch1("path")

    def test_external_group_cannot_back_inference(
        self, pv2_train, unregistered_project, skip_if_no_dlc
    ):
        """The session guard is untouched: no File rows, no inference.

        This is what makes relaxing the import safe -- an externally-sourced
        model still cannot produce a `PoseV2` entry from unregistered data.
        """
        from spyglass.position.v2.video import VidFileGroup

        key = pv2_train.Model().load(unregistered_project, external_videos=True)
        gid = (pv2_train.Model & key).fetch1("vid_group_id")
        with pytest.raises(ValueError, match="no video files"):
            VidFileGroup().get_nwb_file(gid)


@skip_if_no_pose
class TestLoadDispatch:
    """Z06 — one entry point for *any* pretrained model.

    The original request was pretrained models generally, not the zoo
    specifically. ``Model.load`` therefore takes a string and decides: an
    existing path is a project import; otherwise it is looked up in the DLC
    zoo catalog. `load_from_dlc_zoo` stays as the explicit form.

    Dispatch is checked without weights; only the tests that build a real row
    need the download.
    """

    def test_zoo_name_is_not_treated_as_a_path(self, pv2_train):
        """A catalog name must not raise FileNotFoundError."""
        with pytest.raises(Exception) as exc:
            pv2_train.Model().load("superanimal_topviewmouse")
        assert not isinstance(exc.value, FileNotFoundError)

    def test_unknown_string_names_both_possibilities(self, pv2_train):
        """Neither a path nor a zoo entry -- say so, and list the catalog."""
        with pytest.raises(FileNotFoundError) as exc:
            pv2_train.Model().load("not_a_path_nor_a_zoo_model")
        msg = str(exc.value)
        assert "not_a_path_nor_a_zoo_model" in msg
        assert "superanimal" in msg  # catalog surfaced to the user

    def test_existing_path_still_imports_as_project(
        self,
        pv2_train,
        dlc_project_config,
        dlc_bootstrapped_session,
        skip_if_no_dlc,
    ):
        """Regression: the path branch is unchanged.

        Needs ``dlc_bootstrapped_session`` -- the project's videos must be
        registered, or this fails on the very constraint this work relaxes
        rather than on dispatch.
        """
        key = pv2_train.Model().load(dlc_project_config)
        assert pv2_train.Model & key

    def test_explicit_zoo_method_exists(self, pv2_train):
        assert hasattr(pv2_train.Model, "load_from_dlc_zoo")

    def test_rejects_unknown_zoo_name(self, pv2_train):
        """Fail on a typo before any download."""
        with pytest.raises(ValueError, match="superanimal"):
            pv2_train.Model().load_from_dlc_zoo("superanimal_notathing")

    def test_backbone_defaults_per_dataset(self, pv2_train):
        """Backbones differ by dataset -- no single hardcoded default works.

        `superanimal_bird` offers only resnet_50; `superanimal_humanbody` only
        rtmpose_x. Picking hrnet_w32 for all four fails for half the catalog.
        """
        from dlclibrary import get_available_models

        for dataset in ZOO_MODELS:
            chosen = pv2_train.Model()._zoo_default_backbone(dataset)
            assert chosen in list(get_available_models(dataset))

    def test_rejects_backbone_not_offered(self, pv2_train):
        with pytest.raises(ValueError, match="rtmpose_x|offers"):
            pv2_train.Model().load_from_dlc_zoo(
                "superanimal_bird", model_name="hrnet_w32"
            )

    @pytest.mark.parametrize("zoo_name", ZOO_MODELS)
    def test_model_id_prefix_preserves_hash(
        self, pv2_train, zoo_name, requires_network
    ):
        """`zoo-` prefix keeps the id under 32 chars with the hash intact.

        The zoo name itself must never be the prefix:
        `superanimal_topviewmouse-2026091` truncates the hash and two models
        would collide on `model_id`.
        """
        requires_network(zoo_name)
        key = pv2_train.Model().load(zoo_name)
        model_id = key["model_id"]
        assert model_id.startswith("zoo-")
        assert len(model_id) <= 32
        assert len(model_id.split("-")[-1]) == 8

    def test_records_name_mode_and_provenance(
        self, pv2_train, requires_network
    ):
        from spyglass.position.v2.video import VidFileGroup

        requires_network("superanimal_topviewmouse")
        key = pv2_train.Model().load("superanimal_topviewmouse")
        row = (pv2_train.Model & key).fetch1()
        assert row["model_name"] == "superanimal_topviewmouse"
        assert row["training_mode"] == "weights_only"

        ext = VidFileGroup.ExternalVideo & {"vid_group_id": row["vid_group_id"]}
        assert len(ext) == 1
        assert "superanimal_topviewmouse" in ext.fetch1("source")

    def test_weights_stored_portably(self, pv2_train, requires_network):
        """`model_path` must not point into the per-env DLC package cache.

        `get_super_animal_snapshot_path` resolves inside the installed
        deeplabcut package, so storing it verbatim yields a path valid only in
        the importing user's conda env -- a shared-database footgun.
        """
        from pathlib import Path

        from spyglass.settings import pose_project_dir

        requires_network("superanimal_topviewmouse")
        key = pv2_train.Model().load("superanimal_topviewmouse")
        stored = (pv2_train.Model & key).fetch1("model_path")
        assert "site-packages" not in stored
        assert not Path(stored).is_absolute() or str(pose_project_dir) in str(
            stored
        )

    def test_skeleton_parts_marked_imported(
        self, pv2_train, bodypart, requires_network
    ):
        """Zoo vocabulary must not silently widen the curated list (Z08)."""
        requires_network("superanimal_topviewmouse")
        pv2_train.Model().load("superanimal_topviewmouse")
        sources = set((bodypart & {"bodypart": "mouse_center"}).fetch("source"))
        assert sources == {"imported"}

    def test_tf_backed_zoo_model_is_inference_only(self, pv2_train):
        """`dlcrnet` is a TensorFlow backend; DLC-TF continuation is won't-do.

        Such a model must import as `inference_only` rather than let the user
        discover the dead end at train time.
        """
        mode = pv2_train.Model()._zoo_training_mode("dlcrnet")
        assert mode == "inference_only"

    def test_second_import_not_blocked_by_reuse_guard(
        self, pv2_train, requires_network
    ):
        """`ModelSelection` blocks redundant models per skeleton (§3.8)."""
        requires_network("superanimal_topviewmouse")
        pv2_train.Model().load("superanimal_topviewmouse")
        pv2_train.Model().load(
            "superanimal_topviewmouse", allow_redundant_model=True
        )
