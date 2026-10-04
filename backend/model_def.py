import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. The PCSA (Pyramid Channel and Spatial Attention) Module
# ---------------------------------------------------------
class PCSAModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(PCSAModule, self).__init__()
        # Channel Attention (Pyramidal Pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        # Spatial Attention (Multiscale Convolutions)
        # Using a 3x3 and a 5x5 convolution to capture pyramidal spatial context
        self.conv_spatial_3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv_spatial_5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()

        # --- Channel Attention ---
        y_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        y_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_weights = torch.sigmoid(y_avg + y_max)
        x_out = x * channel_weights

        # --- Spatial Attention ---
        max_out, _ = torch.max(x_out, dim=1, keepdim=True)
        avg_out = torch.mean(x_out, dim=1, keepdim=True)
        spatial_in = torch.cat([max_out, avg_out], dim=1)

        spatial_weights_3 = self.conv_spatial_3(spatial_in)
        spatial_weights_5 = self.conv_spatial_5(spatial_in)

        # Fusing the pyramidal spatial features
        spatial_weights = torch.sigmoid(spatial_weights_3 + spatial_weights_5)

        return x_out * spatial_weights

# ---------------------------------------------------------
# 2. The Custom KidneyNeXt Block (with PCSA Injection)
# ---------------------------------------------------------
class PCSA_KidneyNeXt_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PCSA_KidneyNeXt_Block, self).__init__()

        # The 4 parallel pathways from KidneyNeXt
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        # Group Convolutions
        self.g_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.g_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels, bias=False)

        # Channel compression to combine the 4 branches
        concat_channels = in_channels * 4
        self.compress_conv = nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        # *** INJECTING THE NOVEL PCSA MODULE ***
        self.pcsa = PCSAModule(out_channels)

        # Shortcut for residual connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        p_max = self.max_pool(x)
        p_avg = self.avg_pool(x)
        g1 = self.g_conv1(x)
        g2 = self.g_conv2(x)

        # Concatenate features from all 4 branches
        out = torch.cat([p_max, p_avg, g1, g2], dim=1)
        out = self.compress_conv(out)
        out = self.bn(out)
        out = self.gelu(out)

        # Apply the Pyramid Channel & Spatial Attention
        out = self.pcsa(out)

        # Add residual connection
        out = out + self.shortcut(x)
        return out

# ---------------------------------------------------------
# 3. The Complete PCSA-KidneyNeXt Architecture
# ---------------------------------------------------------
class PCSA_KidneyNeXt(nn.Module):
    def __init__(self, num_classes=4):
        super(PCSA_KidneyNeXt, self).__init__()

        # Stem Layer (Extracts initial base features)
        self.stem_conv1 = nn.Conv2d(3, 96, kernel_size=4, stride=4, padding=0, bias=False)
        self.stem_conv2 = nn.Conv2d(3, 96, kernel_size=4, stride=4, padding=0, bias=False)
        self.stem_bn = nn.BatchNorm2d(96)
        self.stem_gelu = nn.GELU()

        # Hierarchical Stages
        self.stage1 = PCSA_KidneyNeXt_Block(in_channels=96, out_channels=96, stride=1)
        self.stage2 = PCSA_KidneyNeXt_Block(in_channels=96, out_channels=192, stride=2)
        self.stage3 = PCSA_KidneyNeXt_Block(in_channels=192, out_channels=384, stride=2)
        self.stage4 = PCSA_KidneyNeXt_Block(in_channels=384, out_channels=768, stride=2)

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # Stem
        s1 = self.stem_conv1(x)
        s2 = self.stem_conv2(x)
        out = (s1 + s2) / 2.0  # Averaging the two parallel initial convolutions
        out = self.stem_gelu(self.stem_bn(out))

        # Stages
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # Head
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# ---------------------------------------------------------
# 4. Initialize and Test the Model
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PCSA_KidneyNeXt(num_classes=4).to(device)

# Pass a dummy tensor to ensure there are no shape mismatch errors
dummy_input = torch.randn(1, 3, 224, 224).to(device)
output = model(dummy_input)

# Calculate total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model initialized on: {device}")
print(f"Output shape (Batch Size, Classes): {output.shape}")
print(f"Total Trainable Parameters: {total_params:,}")

