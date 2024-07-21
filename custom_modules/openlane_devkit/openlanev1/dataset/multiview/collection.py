# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

from openlanev1.io import io
from os.path import join, expanduser

from .frame import MultiViewFrame


class MultiViewCollection:
    r"""
    A collection of frames.
    Waymo multi view frames
    It is designed to support various waymo labels.

    """

    def __init__(self, data_root: str, collection: str) -> None:
        """construct a collection of frames

        Args:
            data_root (str): where the openlane_format is located
            collection (str): name of the pkl file

        Raises:
            FileNotFoundError: if not found the pkl file
        """
        try:
            metas = io.pickle_load(f"{join(expanduser(data_root), collection)}.pkl")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Please collect data first to generate pickle file of the collection."
            )

        self.metas = metas
        self.frames = {k: MultiViewFrame(data_root, v) for k, v in metas.items()}
        self.keys = list(self.frames.keys())

    def get_frame_via_identifier(self, identifier: tuple) -> MultiViewFrame:
        r"""
        Returns a frame with the given identifier (segment_id, timestamp).

        Parameters
        ----------
        identifier : tuple
            (segment_id, timestamp).

        Returns
        -------
        Frame
            A frame identified by the identifier.

        """
        return self.frames[identifier]

    def get_frame_via_index(self, index: int) -> (tuple, MultiViewFrame):
        r"""
        Returns a frame with the given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        (tuple, Frame)
            The identifier of the frame and the frame.

        """
        return self.keys[index], self.frames[self.keys[index]]
