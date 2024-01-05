from src.main import NeuralNetwork as nn

learning_rates = [
        # 0.0000000000000000000000000000000000000000000000000000000000000001,
        # 0.000000000000000000000000000000001,
        # 0.00000000000000001,
        # 0.0000000001,
        # 0.0000001,
        # 0.0001,
        # 0.001,
        # 0.02,
        # 0.025,
        0.03,
        # 0.1,
        # 0.9,
    ]
epochs_list = [
    # 1,
    # 2,
    # 10,

    20,
    # 100,
    #    1000
]
hidden_sizes = [
    # 1,
    # 2,
    # 28,
    784,
    # 21952,
]

batch_rates = [
    # 0.0001,
    0.0005,
    # 0.001,
    #    0.01,
    # 0.1,
    #    0.2,
    # 0.5,
    # 1.0,
]

loss_factor_exponents = [
    # 1.1,
    # 1.2,
    # 1.3,
    # 1.4,
    #  1.5,
    # 1.8,
    # 2.0,
    3.0,
    # 0.1,
    # 0.5,
    # 1.0,
    # 4.0,
    # 5.0,
    # 10.0,
    # 100.0,
]

# for learning_rate_adaptation in learning_rate_adaptations:
#     print("learning_rate_adaptation : ", learning_rate_adaptation)
for loss_factor_exponent in loss_factor_exponents:
    nn.loss_factor = loss_factor_exponent
    for batch_rate in batch_rates:
        results = nn.test_combinations(
            X,
            y,
            # X_test,
            # y_test,
            learning_rates=learning_rates,
            # learning_rate_adaptation=learning_rate_adaptation,
            loss_factor=loss_factor_exponent,
            epochs_list=epochs_list,
            hidden_sizes=hidden_sizes,
            # batch_rates=batch_rates,
            batch_rate=batch_rate,
            # weights_random_samples=10,
            weights_random_samples=1,
            show_training_progress=False,
            # show_training_progress=True,
            # weights_save_path="src/weights/weights_",
        )
        nn.plot_results(results)
    wait = input("close results ? (y/n) : ")
