import numpy as np
from tqdm import tqdm


from tensorslow.datasets import MNIST
from tensorslow.models import ts_mnist_classifier
from tensorslow.optimizers import SGD

data = MNIST(load_test=True, batch_size=128)
x_train, y_train = data.get_train_data()
x_test, y_test = data.get_test_data()
epochs = 10
display_batch = 10
save_model = 100
model = ts_mnist_classifier()
save_loss = 50
model.summary()

opt = SGD(model, learning_rate=5e-4)
train_loss, test_loss = [], []
for epoch in range(epochs):
    print(f'epoch #{epoch+1} out of {epochs}')
    incr = 0
    for data in tqdm(zip(x_train, y_train), desc='training mnist model', total=len(y_train)):
        x, y = data
        batch_loss, grad = model(x, y)
        model.backward(grad)
        opt.update()
        if incr % display_batch == 0:
            print(f'\n{batch_loss:.8f} : this is the loss for the current batch')
        if incr % save_model == 0:
            model.save(f'ts_mnist_sgd_e{epoch}_{incr}.pkl')
        if incr % save_loss == 0:
            train_loss.append(batch_loss)
            loss = 0  
            for x, y in tqdm(zip(x_test, y_test), desc='testing on test data', total=len(x_test)):
                batch_loss, _ = model(x, y)
                loss += batch_loss 
            loss /= len(x_test)
            test_loss.append(loss)
            print(batch_loss, loss, 'train', 'test')
        incr += 1
    
     

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
np.save('train.npy', train_loss)
np.save('test.npy', test_loss)
model.save('ts_mnist_classifier.pkl')


