
def quantize(x, levels):
    """ Quantize x to the nearest of the specified levels. """
    return levels[torch.argmin(torch.abs(levels[:, None] - x), dim=0)]

def straight_through_quantize(x, levels):
    """ Quantize x during forward pass, but use STE in backward pass. """
    x_quantized = quantize(x, levels)
    return x + (x_quantized - x).detach()  # STE: gradients pass through as if quantization hadn't occurred

# Example usage:
x = torch.tensor([0.1, 0.5, 1.5, 1.9], requires_grad=True)
levels = torch.tensor([0.0, 1.0, 2.0])  # Quantization levels
x_quantized = straight_through_quantize(x, levels)

# Compute gradients
x_quantized.sum().backward()
print("Gradients on x:", x.grad)