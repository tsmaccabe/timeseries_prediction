using Main.Attractors3Channel

chunk_size = 100000
num_chunks = 1

selected_system_names = ["genesio_tesi"]
Attractors3Channel.save_selected_system_data(selected_system_names, chunk_size, num_chunks)
