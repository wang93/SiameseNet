import scipy.io as scio


def get_market_attributes(set_name='train'):
    if set_name is not 'train':
        raise NotImplementedError

    path = '../Market-1501_Attribute/market_attribute.mat'
    data = scio.loadmat(path)
    data = data['market_attribute'][0][0]
    train_data = data[set_name]

    attributes = dict()
    fields = ['age', 'backpack', 'bag', 'handbag', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen',
              'downpink', 'downpurple', 'downwhite', 'downyellow', 'upblack', 'upblue', 'upgreen', 'upgray',
              'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down', 'up', 'hair', 'hat', 'gender',
              'image_index']

    for field in fields:
        attributes[field] = train_data[field][0][0][0].tolist()

    attributes['image_index'] = [s[0] for s in attributes['image_index']]
