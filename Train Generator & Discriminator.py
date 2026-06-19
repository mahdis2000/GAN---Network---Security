# ----------------------
    # Train Discriminator
    # ----------------------
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_samples = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ----------------------
    # Train Generator
    # ----------------------
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    d_losses.append(float(d_loss))
    g_losses.append(float(g_loss))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {float(d_loss):.4f} | G Loss: {float(g_loss):.4f}")
