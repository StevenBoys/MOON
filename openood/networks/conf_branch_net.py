import torch.nn as nn


class ConfBranchNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ConfBranchNet, self).__init__()

        self.backbone = backbone

        self.fc = nn.Linear(backbone.feature_size, num_classes)
        self.confidence = nn.Linear(backbone.feature_size, 1)

    # test conf
    def forward(self, x, return_confidence=False):

        _, feature = self.backbone(x, return_feature=True)

        pred = self.fc(feature)
        confidence = self.confidence(feature)

        if return_confidence:
            return pred, confidence
        else:
            return pred
