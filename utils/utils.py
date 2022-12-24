from typing import List
import math

from config import ModelParameters


class Utilities:
    """This class contains utility functions that are used in the project."""

    @staticmethod
    def get_output_sizes(
        res: int = ModelParameters.RESOLUTION,
        conv_kernel_size: int = ModelParameters.CONV_KERNEL_SIZE,
        conv_padding_size: int = ModelParameters.CONV_PADDING_SIZE,
        conv_stride: int = ModelParameters.CONV_STRIDE,
        mp_kernel_size: int = ModelParameters.MP_KERNEL_SIZE,
        mp_padding: int = ModelParameters.MP_PADDING_SIZE,
        mp_stride_length: int = ModelParameters.MP_STRIDE_LENGTH,
    ) -> List[int]:
        """
        This method is used to get the dimension of the output of each layer in the CNN. Configured to match the CNN architecture of the repo.


        Args:
            res (int, optional): input image resolution. Defaults to ModelParameters.RESOLUTION.
            conv_kernel_size (int, optional): Kernel size of the convolutional layer. Defaults to ModelParameters.CONV_KERNEL_SIZE.
            conv_padding_size (int, optional): Padding size of the convolutional layer. Defaults to ModelParameters.CONV_PADDING_SIZE.
            conv_stride (int, optional): Stride length of the convolutional layer. Defaults to ModelParameters.CONV_STRIDE.
            mp_kernel_size (int, optional): Kernel size of the Max Pooling layer. Defaults to ModelParameters.MP_KERNEL_SIZE.
            mp_padding (int, optional): Padding size of the Max Pooling layer. Defaults to ModelParameters.MP_PADDING_SIZE.
            mp_stride_length (int, optional): Stride length of the Max Pooling layer. Defaults to ModelParameters.MP_STRIDE_LENGTH.

        Returns:
            List[int]: List of the output dimensions for each layer in the CNN that are needed.
        """

        def get_layer_output_size(
            input_size: int, filter_size: int, padding: int, stride: int
        ) -> int:
            """
            Helper function to calculate the output size of a layer.


            Returns:
                int: Output size of the layer.
            """

            formula: int = math.floor(
                (((input_size - filter_size + 2 * padding) / stride) + 1)
            )

            return formula

        # Output size from applying conv1 to input
        conv1: int = get_layer_output_size(
            res, conv_kernel_size, conv_padding_size, conv_stride
        )

        # Output size from applying max pooling 1 to conv1
        mp1: int = get_layer_output_size(
            conv1, mp_kernel_size, mp_padding, mp_stride_length
        )

        # Output size from applying conv2 to max pooling 1
        conv2: int = get_layer_output_size(
            mp1, conv_kernel_size, conv_padding_size, conv_stride
        )

        # Output size from applying conv3 to conv 2
        conv3: int = get_layer_output_size(
            conv2, conv_kernel_size, conv_padding_size, conv_stride
        )

        # Output size from applying max pooling 2 to conv3
        mp2: int = get_layer_output_size(
            conv3, mp_kernel_size, mp_padding, mp_stride_length
        )

        # Output size from applying conv 4 to max pooling 2
        conv4: int = get_layer_output_size(
            mp2, conv_kernel_size, conv_padding_size, conv_stride
        )

        # Output size from applying conv5 to conv 4
        conv5: int = get_layer_output_size(
            conv4, conv_kernel_size, conv_padding_size, conv_stride
        )

        # Output size from applying max pooling 3 to conv 5
        mp3: int = get_layer_output_size(
            conv5, mp_kernel_size, mp_padding, mp_stride_length
        )

        # List of dimensions of outputs from each layer that I need.
        outputs_i_need: List[int] = [mp1, conv2, mp2, conv4, mp3]

        return outputs_i_need
