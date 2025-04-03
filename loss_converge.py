import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(range(len(loss_history)), loss_history, label="Total Loss", color="b")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.yscale("log")  # Log scale is often useful for PINN loss plots
plt.title("Loss Convergence During Training")
plt.legend()
plt.grid(True)
plt.show()
