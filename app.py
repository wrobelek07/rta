from flask import Flask


app = Flask(__name__)

@app.route('/')
def hello():

    class Perceptron():
    
        def __init__(self, eta=0.01, n_iter=10):
            self.eta = eta
            self.n_iter = n_iter

        def fit(self, X, y):
            self.w_ = np.zeros(1+X.shape[1])
            self.errors_ = []

            for _ in range(self.n_iter):
                errors = 0
                for xi, target in zip(X,y):
                    update = self.eta*(target-self.predict(xi))
                    self.w_[1:] += update*xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                self.errors_.append(errors)
            return self

        def net_input(self, X):
            return np.dot(X, self.w_[1:])+self.w_[0]

        def predict(self, X):
            return np.where(self.net_input(X)>=0.0,1,-1)
    
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np
    
    iris = load_iris()

    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])

    X = df.iloc[:100,[0,2]].values
    y = df.iloc[0:100,4].values
    y = np.where(y == 0, -1, 1)

    global ppn
    ppn = Perceptron(n_iter=20)
    ppn.fit(X,y)


    return '<h1>Julia Wróbel, 123080</h1>'

@app.route('/predict', methods=['GET'])
def get_prediction():

    from flask import request
    from flask import jsonify
        
    s_l = float(request.args.get('s_l'))
    p_l = float(request.args.get('p_l'))

    data = [s_l, p_l]

    try:
    
        predicted_class = int(ppn.predict(data))

        return jsonify(data=data, predicted_class=predicted_class)
    
    except:
        return jsonify(data=data, predicted_class='Invalid predition')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)