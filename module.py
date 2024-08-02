import torch
import torch.nn as nn

class Web(nn.Module):
  def __init__(self, input_size, output_size, node_num, synapse_per_node, distance, distance_limit, decay, dim, learning):
    self.neurons=nn.ModuleList()
    self.input_size=input_size
    self.output_size=output_size
    for _ in range(node_num):
      self.neurons.append(Neuron(dim, synapse_per_node))
      
  def dist_mask(self): # Reduce computational cost.
    
  def step(self): # Step 1 timestep. Activate all Neurons, and exchange signals.
    buffer=[]
    for nr in self.neurons:
      buffer.append(nr.activate()) # Append synapse state and synapse position.
    for buff in buffer:
      for nr in self.neurons:
        nr.step(buff) # Send signal to all other neurons. If you got memory issues, use dist_mask().
      
  def set_input(self, data): # Set input to input node.
    for i in range(input_size):
      self.neurons[i].sets(data[0])
      
  def get_output(self): # Get output from output node.
  def update(self): # Update synapse position and weight. Oftenly update head position. It is caused by relationship between synapse state and target head state.
    
class Neuron(nn.Module):
  def __init__(self, dim, synapse_num):
    self.synapse_num=synapse_num
    self.decay=nn.Parameter(torch.randn(1))
    self.Hpos=nn.Parameter(torch.randn(dim))
    self.Spos=nn.Parameter(torch.randn(synapse_num, dim))
    self.Sw=nn.Parameter(torch.randn(synapse_num))
    self.dim=dim
    
    self.Hstat=torch.zeros(0)
  def sets(self, sign):
    self.Hstat=sign  
  def step(self, sign, sign_pos):
    lensq=torch.zeros(1)
    for i in range(self.dim):
        lensq+=torch.square(self.pos[i]-sign_pos[i])
    sign=torch.div(sign, lensq)

    self.Hstat=self.Hstat*self.decay + sign
    self.Sstat=self.Hstat*self.Sw
    
  def activate(self):
    return self.
