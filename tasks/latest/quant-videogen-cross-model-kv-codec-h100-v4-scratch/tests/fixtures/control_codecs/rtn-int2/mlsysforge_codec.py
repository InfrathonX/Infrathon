"""Paper baseline RTN-INT2 (QVG eq. 1-2): symmetric per-group round-to-nearest, group 16."""
from codec.rtn_common import decode_one, encode_one


def encode(k, v):
    return encode_one(k, 2), encode_one(v, 2)


def decode(packed_k, packed_v):
    return decode_one(packed_k), decode_one(packed_v)
