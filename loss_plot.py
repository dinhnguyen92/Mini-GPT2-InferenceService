import matplotlib.pyplot as plt
import io

def generate_loss_plot(train_losses, test_losses):
    plt.style.use('dark_background')

    # Plot train and test losses
    plt.plot(train_losses, label='Train Loss', color='#0c5a94')
    plt.plot(test_losses, label='Test Loss', color='#6dffff')

    # Add labels and title
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Train and Test Losses', fontsize=16)

    # Show legend
    plt.legend(fontsize='large')

    # Save the plot to an in-memory binary stream
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')

    # Reset the buffer's cursor to the beginning
    buffer.seek(0)

    # Access the binary data from the stream
    plot_data = buffer.getvalue()

    # Close the plot to avoid displaying it in the notebook
    plt.close()
    buffer.close()

    return plot_data
