#!/usr/bin/env python
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import argparse
import numpy as np
from PIL import Image
from builtins import range
from functools import partial
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config

if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

from .utilities import loggingConfig
log = loggingConfig('common.tritonclient')

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

class TritonClient:
    def __init__(self, config: dict):
        """[summary]

        Args:
            config (dict): [description]

        Raises:
            Exception: [description]
        """
        # get the instance variables from the supplied
        # configuration dictionary. Set default values
        # if keys are not present in the config parameter
        self.model_name = config.get('model-name', 'r2plus1d_32')
        self.model_version = config.get('model-version', 1)
        protocol = config.get('protocol', 'gRPC')
        self.protocol = ProtocolType.from_str(protocol)
        self.url = config.get('url', 'localhost:8001')
        self.async_set = config.get('async-set', False)
        self.streaming = config.get('streaming', True)
        self.batch_size = config.get('batch-size', 1)
        self.verbose = config.get('verbose', False)
        self.scaling = config.get('scaling', None)
        self.classes = config.get('classes', 1)

        if self.streaming and self.protocol != ProtocolType.GRPC:
            raise Exception("Streaming is only allowed with gRPC protocol")

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self.input_name, self.output_name, self.c, self.h, self.w, self.t, self.format, self.dtype = self.parse_model(
                                                                                                            self.url, 
                                                                                                            self.protocol, 
                                                                                                            self.model_name, 
                                                                                                            self.batch_size, 
                                                                                                            self.verbose
                                                                                                            )
        
        # intialize inferContext
        self.ctx = InferContext(
                    self.url, 
                    self.protocol, 
                    self.model_name,
                    self.model_version, 
                    self.verbose, 
                    0, 
                    self.streaming
                    )
        self.results = list()
        self.image_data = list()
        self.sent_count = 0
        self.processed_count = 0
        self.user_data = UserData()

    def get_result(self, image: np.array):
        """[summary]

        Args:
            image (np.array): [description]

        Returns:
            [type]: [description]
        """
        img = Image.fromarray(image)
        if len(self.image_data) == (self.t * self.batch_size):
            input_batch = self.prepare_batch(self.image_data)
            self.results = self.send_request(input_batch)
        else:    
            self.image_data.append(self.preprocess(img))

        return self.results

    def prepare_batch(self, data: list):
        """[summary]

        Args:
            data (list): [description]

        Returns:
            [type]: [description]
        """
        input_batch = list()
        for batch_num in range(self.batch_size):
            input_data = np.asarray([data[(batch_num * self.t):((batch_num + 1) * self.t)]])
            input_data = self.reshape(input_data)
            input_batch.append(input_data)
        
        self.image_data = list()

        return input_batch

    def get_result_str(self) -> str:
        """Processes a list containing results from a video action recognition
        model and returns the information as a string.

        Args:
            clip_class (list): list of tuples containing inference results

        Returns:
            str: [description]
        """
        result_str = ''
        for cls in self.results:
            if len(cls) == 3:
                result_str += '{}: {}%\n'.format(cls[2], int(max(0, cls[1]) * 100))
            
        return result_str

    def send_request(self, input_batch: list) -> list:
        """[summary]

        Args:user_data
            list: [description]
        """
        results = list()

        try:
            if not self.async_set:
                results.append(self.ctx.run(
                    { self.input_name : input_batch },
                    { self.output_name : (InferContext.ResultFormat.CLASS, self.classes) },
                    self.batch_size))
            else:
                self.ctx.async_run(partial(self.completion_callback, self.user_data),
                                { self.input_name :input_batch },
                                { self.output_name : (InferContext.ResultFormat.CLASS, self.classes) },
                                self.batch_size)
                self.sent_count += 1
                # For async, retrieve results according to the send order
                while self.processed_count < self.sent_count:
                    request_id = self.user_data._completed_requests.get()
                    results.append(self.ctx.get_async_run_results(request_id))
                    self.processed_count += 1
        except Exception as e:
            print(str(e))
        
        return self.postprocess(results)

    def reshape(self, data):
        return np.reshape(data, (self.c, self.h, self.w, self.t))

    # Callback function used for async_run()
    def completion_callback(self, user_data, infer_ctx, request_id):
        user_data._completed_requests.put(request_id)

    def model_dtype_to_np(self, model_dtype):
        """[summary]

        Args:
            model_dtype ([type]): [description]

        Returns:
            [type]: [description]
        """
        if model_dtype == model_config.TYPE_BOOL:
            return np.bool
        elif model_dtype == model_config.TYPE_INT8:
            return np.int8
        elif model_dtype == model_config.TYPE_INT16:
            return np.int16
        elif model_dtype == model_config.TYPE_INT32:
            return np.int32
        elif model_dtype == model_config.TYPE_INT64:
            return np.int64
        elif model_dtype == model_config.TYPE_UINT8:
            return np.uint8
        elif model_dtype == model_config.TYPE_UINT16:
            return np.uint16
        elif model_dtype == model_config.TYPE_FP16:
            return np.float16
        elif model_dtype == model_config.TYPE_FP32:
            return np.float32
        elif model_dtype == model_config.TYPE_FP64:
            return np.float64
        elif model_dtype == model_config.TYPE_STRING:
            return np.dtype(object)
        return None

    def parse_model(self, url, protocol, model_name, batch_size, verbose=False):
        """Check the configuration of a model to make sure it meets the
        requirements for a video classification network (as expected by
        this client)

        Args:
            url ([type]): [description]
            protocol ([type]): [description]
            model_name ([type]): [description]
            batch_size ([type]): [description]
            verbose (bool, optional): [description]. Defaults to False.

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]

        Returns:
            [type]: [description]
        """
        self.ctx = ServerStatusContext(url, protocol, model_name, verbose)
        server_status = self.ctx.get_server_status()

        if model_name not in server_status.model_status:
            raise Exception("unable to get status for '" + model_name + "'")

        status = server_status.model_status[model_name]
        config = status.config

        if len(config.input) != 1:
            raise Exception("expecting 1 input, got {}".format(len(config.input)))
        if len(config.output) != 1:
            raise Exception("expecting 1 output, got {}".format(len(config.output)))

        input = config.input[0]
        output = config.output[0]

        if output.data_type != model_config.TYPE_FP32:
            raise Exception("expecting output datatype to be TYPE_FP32, model '" +
                            model_name + "' output type is " +
                            model_config.DataType.Name(output.data_type))

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Variable-size dimensions are not
        # currently supported.
        non_one_cnt = 0
        for dim in output.dims:
            if dim == -1:
                raise Exception("variable-size dimension in model output not supported")
            if dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model specifying maximum batch size of 0 indicates that batching
        # is not supported and so the input tensors do not expect an "N"
        # dimension (and 'batch_size' should be 1 so that only a single
        # image instance is inferred at a time).
        max_batch_size = config.max_batch_size
        if max_batch_size == 0:
            if batch_size != 1:
                raise Exception("batching not supported for model '" + model_name + "'")
        else: # max_batch_size > 0
            if batch_size > max_batch_size:
                raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, model_name))

        # Model input must have 4 dims
        if len(input.dims) != 4:
            raise Exception(
                "expecting input to have 4 dimensions, model '{}' input has {}".format(
                    model_name, len(input.dims)))

        # Variable-size dimensions are not currently supported.
        for dim in input.dims:
            if dim == -1:
                raise Exception("variable-size dimension in model input not supported")

        if (input.format != model_config.ModelInput.FORMAT_NHWC or 
            input.format != model_config.ModelInput.FORMAT_NCHW):
            c = input.dims[0]
            h = input.dims[1]
            w = input.dims[2]
            t = input.dims[3]
            print(h)
        else:
            raise Exception("expecting input to be NONE. Got {}".format(input.format))


        return (input.name, output.name, c, h, w, t, input.format, self.model_dtype_to_np(input.data_type))

    def preprocess(self, img):
        """Pre-process an image to meet the size, type and format
        requirements specified by the parameters.

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        #np.set_printoptions(threshold='nan')

        if self.c == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        resized_img = sample_img.resize((self.w, self.h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:,:,np.newaxis]

        typed = resized.astype(self.dtype)

        if self.scaling == 'INCEPTION':
            scaled = (typed / 128) - 1
        elif self.scaling == 'VGG':
            if self.c == 1:
                scaled = typed - np.asarray((128,), dtype=self.dtype)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=self.dtype)
        else:
            scaled = typed

        # Swap to CHW if necessary
        if self.format == model_config.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered

    def postprocess(self, results: list) -> list:
        """Post-process results to show classifications.

        Args:
            results (list): [description]

        Returns:
            list: [description]
        """
        if len(results[0]) != 1:
            log.error("expected 1 result, got {}".format(len(results[0])))
            return

        batched_result = list(results[0].values())[0]
        if len(batched_result) != self.batch_size:
            log.warn("expected {} results, got {}".format(self.batch_size, len(batched_result)))
            return

        if len(batched_result) > 1:
            def get_column(arr, index):
                """[summary]

                Args:
                    arr ([type]): [description]
                    index ([type]): [description]

                Returns:
                    [type]: [description]
                """
                return [row[index] for row in arr]

            # average the prediction scores of each class
            avg_result = list()
            for idx in range(len(batched_result[0])):
                # get a column from batched_results containing results
                # for a sing class
                class_result = get_column(batched_result, idx)

                # calculate the average of the predicition scores
                avg_score = sum(value[1] for value in class_result) / len(class_result)
                
                # create a new list with one entry per class
                # where the prediction score is the avg_score
                class_id = batched_result[0][idx][0]
                class_name = batched_result[0][idx][2]
                avg_result.append((class_id, avg_score, class_name))
            return avg_result

        else:
            return batched_result[0]