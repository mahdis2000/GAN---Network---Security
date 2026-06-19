latent_dim = 32
epochs = 300
batch_size = 128
lr = 0.0002

n_features = X_train.shape[1]

print("Latent Dim:", latent_dim)
print("Epochs:", epochs)
print("Features:", n_features)

# ==============================
# GENERATOR
# ==============================

generator = Sequential([
    Dense(64, input_dim=latent_dim),
    LeakyReLU(negative_slope=0.2),

    Dense(128),
    LeakyReLU(negative_slope=0.2),

    Dense(256),
    LeakyReLU(negative_slope=0.2),

    Dense(n_features, activation='sigmoid')
])

# ==============================
# DISCRIMINATOR
# ==============================

discriminator = Sequential([
    Dense(256, input_dim=n_features),
    LeakyReLU(negative_slope=0.2),

    Dense(128),
    LeakyReLU(negative_slope=0.2),

    Dense(64),
    LeakyReLU(negative_slope=0.2),

    Dense(1, activation='sigmoid')
])

opt_d = Adam(learning_rate=lr, beta_1=0.5)

discriminator.compile(loss='binary_crossentropy', optimizer=opt_d)
