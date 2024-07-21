# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

# this implementation is based on the original implementation of openlanev2

from ..io import io
from os.path import join, expanduser

from .frame import Frame


class Collection:
    r"""
    A collection of frames.

    """

    def __init__(self, data_root: str, meta_root: str, collection: str) -> None:
        r"""
        Parameters
        ----------
        data_root : str
        meta_root : str
        collection : str
            Name of collection.

        """
        try:
            meta = io.pickle_load(f"{join(expanduser(meta_root), collection)}.pkl")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Please run the generate_pkl preprocessing first to generate pickle file of the collection."
            )

        self.meta = meta
        self.frames = {k: Frame(data_root, v) for k, v in meta.items()}
        self.keys = list(self.frames.keys())

    def get_frame_via_identifier(self, identifier: tuple) -> Frame:
        r"""
        Returns a frame with the given identifier (split, segment_id, timestamp).

        Parameters
        ----------
        identifier : tuple
            (split, segment_id, timestamp).

        Returns
        -------
        Frame
            A frame identified by the identifier.

        """
        return self.frames[identifier]

    def get_frame_via_index(self, index: int) -> (tuple, Frame):
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
