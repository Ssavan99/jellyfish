# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: predict.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'predict.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rpredict.proto\x12\rmodel_serving\" \n\x0b\x43lientToken\x12\x11\n\tclient_id\x18\x01 \x01(\x04\"P\n\nClientInfo\x12\x0b\n\x03slo\x18\x01 \x01(\x04\x12\x12\n\nframe_rate\x18\x02 \x01(\x04\x12\x0f\n\x07init_bw\x18\x03 \x01(\x02\x12\x10\n\x08lat_wire\x18\x04 \x01(\x02\"A\n\x0fRegisterRequest\x12.\n\x0b\x63lient_info\x18\x01 \x01(\x0b\x32\x19.model_serving.ClientInfo\"Z\n\x10RegisterResponse\x12\x30\n\x0c\x63lient_token\x18\x01 \x01(\x0b\x32\x1a.model_serving.ClientToken\x12\x14\n\x0cmodel_number\x18\x02 \x01(\x03\"X\n\tFrameMeta\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x04\x12\x15\n\rdesired_model\x18\x02 \x01(\x03\x12\x16\n\x0esend_timestamp\x18\x03 \x01(\x01\x12\n\n\x02\x62w\x18\x04 \x01(\x01\"\x18\n\tFrameData\x12\x0b\n\x03img\x18\x01 \x01(\x0c\"O\n\rDetectionMeta\x12\x12\n\nused_model\x18\x01 \x01(\x03\x12\x16\n\x0esend_timestamp\x18\x02 \x01(\x01\x12\x12\n\nnext_model\x18\x03 \x01(\x03\"\x9e\x01\n\x0ePredictRequest\x12\x30\n\x0c\x63lient_token\x18\x01 \x01(\x0b\x32\x1a.model_serving.ClientToken\x12,\n\nframe_meta\x18\x02 \x01(\x0b\x32\x18.model_serving.FrameMeta\x12,\n\nframe_data\x18\x03 \x01(\x0b\x32\x18.model_serving.FrameData\"\xaf\x01\n\x0fPredictResponse\x12\x30\n\x0c\x63lient_token\x18\x01 \x01(\x0b\x32\x1a.model_serving.ClientToken\x12,\n\nframe_meta\x18\x02 \x01(\x0b\x32\x18.model_serving.FrameMeta\x12.\n\x08\x64\x65t_meta\x18\x03 \x01(\x0b\x32\x1c.model_serving.DetectionMeta\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\"\x8f\x01\n\x11PredictRequestAck\x12\x30\n\x0c\x63lient_token\x18\x01 \x01(\x0b\x32\x1a.model_serving.ClientToken\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\x04\x12\x1a\n\x12server_recv_req_ts\x18\x03 \x01(\x01\x12\x1a\n\x12server_send_ack_ts\x18\x04 \x01(\x01\"\x07\n\x05\x45mpty2\xc0\x02\n\x0cModelServing\x12N\n\x07predict\x12\x1d.model_serving.PredictRequest\x1a\x1e.model_serving.PredictResponse\"\x00(\x01\x30\x01\x12O\n\x0bpredict_ack\x12\x1a.model_serving.ClientToken\x1a .model_serving.PredictRequestAck\"\x00\x30\x01\x12M\n\x08register\x12\x1e.model_serving.RegisterRequest\x1a\x1f.model_serving.RegisterResponse\"\x00\x12@\n\nunregister\x12\x1a.model_serving.ClientToken\x1a\x14.model_serving.Empty\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'predict_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CLIENTTOKEN']._serialized_start=32
  _globals['_CLIENTTOKEN']._serialized_end=64
  _globals['_CLIENTINFO']._serialized_start=66
  _globals['_CLIENTINFO']._serialized_end=146
  _globals['_REGISTERREQUEST']._serialized_start=148
  _globals['_REGISTERREQUEST']._serialized_end=213
  _globals['_REGISTERRESPONSE']._serialized_start=215
  _globals['_REGISTERRESPONSE']._serialized_end=305
  _globals['_FRAMEMETA']._serialized_start=307
  _globals['_FRAMEMETA']._serialized_end=395
  _globals['_FRAMEDATA']._serialized_start=397
  _globals['_FRAMEDATA']._serialized_end=421
  _globals['_DETECTIONMETA']._serialized_start=423
  _globals['_DETECTIONMETA']._serialized_end=502
  _globals['_PREDICTREQUEST']._serialized_start=505
  _globals['_PREDICTREQUEST']._serialized_end=663
  _globals['_PREDICTRESPONSE']._serialized_start=666
  _globals['_PREDICTRESPONSE']._serialized_end=841
  _globals['_PREDICTREQUESTACK']._serialized_start=844
  _globals['_PREDICTREQUESTACK']._serialized_end=987
  _globals['_EMPTY']._serialized_start=989
  _globals['_EMPTY']._serialized_end=996
  _globals['_MODELSERVING']._serialized_start=999
  _globals['_MODELSERVING']._serialized_end=1319
# @@protoc_insertion_point(module_scope)
