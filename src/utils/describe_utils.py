import torch as t


def average_cosine_sim_final(model, dataset):
    """
    Compute the average cosine similarity between activations of a layer of a model,
    using the dataset to produce activations.
    """

    for layer in range(num_layers):
      normed_embeddings = model.W_E / model.W_E.norm(dim=1)[:, None]
      sim = t.mm(normed_embeddings, normed_embeddings.t())
      sim = sim.cpu()

      sim_values = sim[t.triu(t.ones_like(sim), diagonal=1) == 1]
      for location in locations:

        location_layer = location.format(layer)
        X = dataset_activations_optimised_new(
            model,
            dataset,
            location_layer,
            2,
            True
        )


        normed_embeddings = X / X.norm(dim=1)[:, None]
        sim = t.mm(normed_embeddings, normed_embeddings.t())
        sim = sim.cpu()

        sim_values = sim[t.triu(t.ones_like(sim), diagonal=1) == 1]

        # Find the average
        average = t.mean(sim_values)
        results[location].append(average.item())

    # Plotting the histogram
    for key, values in results.items():
        plt.plot(values, label=names[key])

    print(results)

    # Adding labels and title
    plt.xlabel('Layer Index', fontsize=16)
    plt.ylabel('Average Cosine Similarity', fontsize=16)
    plt.title(f'{model_name}', fontsize=16)
    plt.legend(fontsize=16)  # Display the legend
    plt.savefig(f'{model_name}-2-cosine-sim.pdf', format='pdf')
    plt.show()