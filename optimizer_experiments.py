import pickle
from tqdm import tqdm


from tensorslow.datasets import MNIST
from tensorslow.models import ts_mnist_classifier
from tensorslow.optimizers import SGD, SGDMomentum, SGDNesterov, RMSProp, ADAM

data = MNIST(load_test=True, batch_size=128)
x_train, y_train = data.get_train_data()
x_test, y_test = data.get_test_data()
epochs = 10
lr = 5e-4
display_batch = 10
save_model = 100
save_loss = 50
test_loss_path = 'test_losses.pkl'
train_loss_path = 'train_losses.pkl'

opt_dict = {
    'sgd': SGD,
    'sgd_momentum': SGDMomentum,
    'sgd_nesterov': SGDNesterov,
    'rms_prop': RMSProp,
    'adam': ADAM,
}

train_losses, test_losses = dict(), dict()

for opt_name, opt in opt_dict.items():
    model = ts_mnist_classifier()
    opt = opt(model, learning_rate=lr)
    model.summary()
    train_losses[opt_name] = []
    test_losses[opt_name] = []
    for epoch in range(epochs):
        print(f'epoch #{epoch+1} out of {epochs}')
        incr = 0
        for data in tqdm(zip(x_train, y_train), desc=f'training mnist model with {opt_name}', total=len(y_train)):
            x, y = data
            batch_loss, grad = model(x, y)
            model.backward(grad)
            opt.update()
            if incr % display_batch == 0:
                print(f'\n{batch_loss:.8f} : this is the loss for the current batch')
            if incr % save_model == 0:
                model.save(f'ts_mnist_{opt_name}_e{epoch}_{incr}.pkl')
            if incr % save_loss == 0:
                train_losses[opt_name].append(batch_loss)
                loss = 0
                for x, y in tqdm(zip(x_test, y_test), desc='testing on test data', total=len(x_test)):
                    batch_loss, _ = model(x, y)
                    loss += batch_loss
                loss /= len(x_test)
                test_losses[opt_name].append(loss)
                print(f'\n{loss:.8f} : this is the avg test loss')
            incr += 1

with open(train_loss_path, 'wb') as fh:
    pickle.dump(train_loss_path, fh)

with open(test_loss_path, 'wb') as fh:
    pickle.dump(test_loss_path, fh)



