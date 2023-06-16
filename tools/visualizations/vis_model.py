# pip install netron(netron)
# netron
import netron

# Visualizing neural networks via netron
modelPath = '../utils/Ml4fTransformerV2.onnx'
# Priming model
netron.start(modelPath)
