"""Lossless control: the payload is a copy of the input."""
import torch


def encode(k, v):
    return {"x": k.contiguous().clone()}, {"x": v.contiguous().clone()}


def decode(packed_k, packed_v):
    return packed_k["x"], packed_v["x"]
