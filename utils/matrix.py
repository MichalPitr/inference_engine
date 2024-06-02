import numpy as np

n = 10
m = 5
k = 1
A = np.random.rand(n, m)
B = np.random.rand(m, k)
C = np.random.rand(n, 1)
out = A@B + C


flattened = [element for row in A for element in row]  
# Convert elements to strings and join them with commas
output = ", ".join(map(str, flattened)) 
print("A:")
print(output)
print()


flattened = [element for row in B for element in row]  
# Convert elements to strings and join them with commas
output = ", ".join(map(str, flattened)) 
print("B:")
print(output)
print()


flattened = [element for row in C for element in row]  
# Convert elements to strings and join them with commas
output = ", ".join(map(str, flattened)) 
print("C:")
print(output)
print()


flattened = [element for row in out for element in row]  
# Convert elements to strings and join them with commas
output = ", ".join(map(str, flattened)) 
print("out:")
print(output)
print()


