import os
from pathlib import Path

import datajoint as dj
import keypoint_moseq as kpms

from spyglass.common import AnalysisNwbfile
from spyglass.position.position_merge import PoseOutput
from spyglass.utils import SpyglassMixin

from .core import PoseGroup, format_dataset_for_moseq, results_to_df

schema = dj.schema("moseq_v1")


@schema
class MoseqModelParams(SpyglassMixin, dj.Manual):
    definition = """
    model_params_name: varchar(80)
    ---
    model_params: longblob
    """


@schema
class MoseqModelSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> PoseGroup
    -> MoseqModelParams
    """
    # relevant parameters:
    # - skeleton
    # - num_ar_iters
    # - kappa
    # - num_epochs
    # - anterior_bodyparts
    # - posterior_bodyparts


@schema
class MoseqModel(SpyglassMixin, dj.Computed):
    definition = """
    -> MoseqModelSelection
    ---
    project_dir: varchar(255)
    epochs_trained: int
    model_name = "": varchar(255)
    """

    def make(self, key):
        """Method to train a model and insert the resulting model into the MoseqModel table

        Parameters
        ----------
        key : dict
            key to a single MoseqModelSelection table entry
        """
        # TODO: make config a method
        # TODO: set kappa during ARHMM
        # TODO: make pca components settable

        model_params = (MoseqModelParams & key).fetch1("model_params")

        # set up the project and config
        project_dir = (
            "/home/sambray/Documents/moseq_test_proj2"  # TODO: make this better
        )
        video_dir = (
            "/home/sambray/Documents/moseq_test_vids2"  # TODO: make this better
        )
        # make symlinks to the videos in a single directory
        os.makedirs(video_dir, exist_ok=True)
        # os.makedirs(project_dir, exist_ok=True)
        video_paths = (PoseGroup & key).fetch_video_paths()
        for video in video_paths:
            destination = os.path.join(video_dir, os.path.basename(video))
            if not os.path.exists(destination):
                os.symlink(video, destination)
        bodyparts = (PoseGroup & key).fetch1("bodyparts")
        skeleton = model_params["skeleton"]
        anterior_bodyparts = model_params.get("anterior_bodyparts", None)
        posterior_bodyparts = model_params.get("posterior_bodyparts", None)

        kpms.setup_project(
            str(project_dir),
            video_dir=str(video_dir),
            bodyparts=bodyparts,
            skeleton=skeleton,
            use_bodyparts=bodyparts,
            anterior_bodyparts=anterior_bodyparts,
            posterior_bodyparts=posterior_bodyparts,
        )

        config = lambda: kpms.load_config(project_dir)

        # fetch the data and format it for moseq
        coordinates, confidences = PoseGroup().fetch_pose_datasets(
            key, format_for_moseq=True
        )
        data, metadata = kpms.format_data(coordinates, confidences, **config())

        # fit pca of data
        pca = kpms.fit_pca(**data, **config())
        kpms.save_pca(pca, project_dir)

        # create the model
        model = kpms.init_model(data, pca=pca, **config())
        # run the autoregressive fit on the model
        num_ar_iters = model_params["num_ar_iters"]
        model, model_name = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            ar_only=True,
            num_iters=num_ar_iters,
        )
        # load model checkpoint
        model, data, metadata, current_iter = kpms.load_checkpoint(
            project_dir, model_name, iteration=num_ar_iters
        )
        # update the hyperparameters
        kappa = model_params["kappa"]
        model = kpms.update_hypparams(model, kappa=kappa)
        # run fitting on the complete model
        num_epochs = model_params["num_epochs"]
        model = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            ar_only=False,
            start_iter=current_iter,
            num_iters=current_iter + num_epochs,
        )[0]
        # reindex syllables by frequency
        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        self.insert1(
            {
                **key,
                "project_dir": project_dir,
                "epochs_trained": num_epochs + num_ar_iters,
                "model_name": model_name,
            }
        )

    def extend_training(self, key: dict, num_epochs: int):
        """Method to run additional training epochs on a model and update the epochs_trained attribute

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def analyze_pca(self, key: dict = None):
        """Method to analyze the PCA of a model

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        if key is None:
            key = {}
        project_dir = (self & key).fetch1("project_dir")
        pca = kpms.load_pca(project_dir)
        config = lambda: kpms.load_config(project_dir)
        kpms.print_dims_to_explain_variance(pca, 0.9)
        kpms.plot_scree(pca, project_dir=project_dir)
        kpms.plot_pcs(pca, project_dir=project_dir, **config())

    def fetch_model(self, key: dict = None):
        """Method to fetch the model from the MoseqModel table

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Returns
        -------
        dict
            model dictionary
        """
        if key is None:
            key = {}
        model = kpms.load_checkpoint(
            (self & key).fetch1("project_dir"),
            (self & key).fetch1("model_name"),
        )[0]
        return model


@schema
class MoseqSyllableSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> PoseOutput.proj(pose_merge_id='merge_id')
    -> MoseqModel
    ---
    num_iters = 500: int
    """

    def insert(self, rows, **kwargs):
        """Override insert to check that the bodyparts in the model are in the data"""
        for row in rows:
            self.validate_bodyparts(row)
        super().insert(rows, **kwargs)

    def validate_bodyparts(self, key):
        if self & key:
            return
        model_bodyparts = (PoseGroup & key).fetch1("bodyparts")
        merge_key = {"merge_id": key["pose_merge_id"]}
        bodyparts_df = (PoseOutput & merge_key).fetch_dataframe()
        data_bodyparts = bodyparts_df.keys().get_level_values(0).unique().values
        for bodypart in model_bodyparts:
            if bodypart not in data_bodyparts:
                raise ValueError(
                    f"Error in row {row}: " + f"Bodypart {bodypart} not in data"
                )


@schema
class MoseqSyllable(SpyglassMixin, dj.Computed):
    definition = """
    -> MoseqSyllableSelection
    ---
    -> AnalysisNwbfile
    moseq_object_id: int
    """

    def make(self, key):
        model = MoseqModel().fetch_model(key)
        project_dir = (MoseqModel & key).fetch1("project_dir")
        # video_dir = (
        #     "/home/sambray/Documents/moseq_test_vids2"  # TODO: make this better
        # )

        merge_key = {"merge_id": key["pose_merge_id"]}
        bodyparts = (PoseGroup & key).fetch1("bodyparts")
        config = lambda: kpms.load_config(project_dir)
        model_name = (MoseqModel & key).fetch1("model_name")
        num_iters = (MoseqSyllableSelection & key).fetch1("num_iters")

        # load data and format for moseq
        video_path = (PoseOutput & merge_key).fetch_video_name()
        video_name = Path(video_path).stem + ".mp4"
        bodyparts_df = (PoseOutput & merge_key).fetch_dataframe()

        if bodyparts is None:
            bodyparts = bodyparts_df.keys().get_level_values(0).unique().values
        datasets = {video_name: bodyparts_df[bodyparts]}
        coordinates, confidences = format_dataset_for_moseq(datasets, bodyparts)
        data, metadata = kpms.format_data(coordinates, confidences, **config())

        # apply model
        results = kpms.apply_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            **config(),
            num_iters=num_iters,
        )

        # format results into dataframe for saving
        results_df = results_to_df(results)
        results_df.index = bodyparts_df.index

        # save results into a nwbfile
        nwb_file_name = PoseOutput.merge_get_parent(merge_key).fetch1(
            "nwb_file_name"
        )
        nwb_file_name = PoseOutput.merge_get_parent(merge_key).fetch1(
            "nwb_file_name"
        )
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        key["analysis_file_name"] = analysis_file_name
        key["moseq_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, results_df.reset_index(), "moseq"
        )
        AnalysisNwbfile().add(nwb_file_name, analysis_file_name)

        self.insert1(key)
