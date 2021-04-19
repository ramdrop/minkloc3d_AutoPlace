# Author: Jacek Komorowski
# Warsaw University of Technology

import models.minkloc as minkloc


def model_factory(params):
    in_channels = 1

    if 'MinkFPN' in params.model_params.model:
        model = minkloc.MinkLoc(params.model_params.model,  # 'MinkFPN_GeM'
                                in_channels=in_channels,    # 1
                                feature_size=params.model_params.feature_size,  # 256
                                output_dim=params.model_params.output_dim,      # 256
                                planes=params.model_params.planes,              # [32, 64, 64]
                                layers=params.model_params.layers,              # [1, 1, 1]
                                num_top_down=params.model_params.num_top_down,  # 1
                                conv0_kernel_size=params.model_params.conv0_kernel_size)    # 5
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    return model
