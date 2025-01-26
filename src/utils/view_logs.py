from tensorboard import program


# Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "../../logs"])
url = tb.launch()
print(f"Tensorflow listening on {url}")

input("Press Enter to continue...")
