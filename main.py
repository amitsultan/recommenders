import matplotlib.pyplot as plt
from models.BPR import *
from models.Metrics import *

#  run a & b should have the same metrics and k_lst sizes
def record_experiment(name, run_a, run_b, title_a, title_b):
    fig, ax = plt.subplots(1, len(list(run_a.keys())))
    index = 0
    for metric in run_a.keys():
        metric_a = run_a[metric]
        metric_b = run_b[metric]
        ax[index].set_title(f'{metric} comparision of models: [{title_a}, {title_b}]')
        ax[index].plot(metric_a['k_lst'], metric_a['values'], label='title_a')
        ax[index].plot(metric_b['k_lst'], metric_b['values'], label='title_b')
        ax[index].set_xlabel('K value')
        ax[index].set_ylabel(f'{metric} value')
        index += 1
    plt.show()

def run_scistarter_experiment(metric_config):
    train = Dataset('datasets/scistarter/')
    train.read_dataset_from_csv('scistarter', 'scistarter_train.csv')
    train.calculate_popularity()
    print('Done loading train dataset + popularity calculation')
    train.calculate_co_count()
    print('Done CoCount calculation')

    test = Dataset('datasets/scistarter/')
    test.read_dataset_from_csv('scistarter', 'scistarter_test.csv')
    # test.calculate_popularity()
    print('Done loading test dataset + popularity calculation')


    #is that correct?
    test.item_statistics = train.item_statistics
    test.popularity = train.popularity
    test.head_items = train.head_items
    test.tail_items = train.tail_items

    bpr = BPR(100, train)
    bpr.SGD(train, 5)
    a = compute_metrics(bpr, test, metric_config)
    print('done SGD')
    print('computing lift')
    b = compute_metrics(bpr, test, metric_config, with_lift=True)
    print('no lift')
    for k, v in a.items():
        print(f'{k}: {v}')
    print('lift')
    for k, v in b.items():
        print(f'{k}: {v}')
    record_experiment('Scistarter', a, b, 'BPR', 'Lift-Boosted BPR')


def run_zooniverse_experiment():
    train = Dataset('datasets/zooniverse/')
    train.read_dataset_from_csv('zooniverse', 'train_jun.csv')
    # train.calculate_popularity()
    print('Done loading train dataset + popularity calculation')
    train.calculate_co_count()
    print('Done CoCount calculation')

    test = Dataset('datasets/zooniverse/')
    test.read_dataset_from_csv('zooniverse', 'test_jun.csv')
    # test.calculate_popularity()
    print('Done loading test dataset + popularity calculation')

    bpr = BPR(100, train)
    bpr.SGD(train, 5)
    a = compute_metrics(bpr, test, {'recall': [1, 2, 3, 5, 10], 'hitrate': [1, 2, 3, 5, 10], 'precision': [1, 2, 3, 5, 10]})
    print('done SGD')
    print('computing lift')
    b = compute_metrics(bpr, test, {'recall': [1, 2, 3, 5, 10], 'hitrate': [1, 2, 3, 5, 10], 'precision': [1, 2, 3, 5, 10]}, with_lift=True)
    print('no lift')
    print(a)
    print('lift')
    print(b)

if __name__ == "__main__":
    metric_config = {'recall': [1, 2, 3, 5, 10],
                     'hitrate': [1, 2, 3, 5, 10],
                     'head_hitrate': [1, 2, 3, 5, 10],
                     'tail_hitrate': [1, 2, 3, 5, 10],
                     'precision': [1, 2, 3, 5, 10]}
    run_scistarter_experiment(metric_config)
    # run_zooniverse_experiment()









# def main():
#     scistarter = 'datasets/scistarter/'
#     ml_1 = 'datasets/zooniverse/'
#     output_data = ml_1 + "zooniverse.pickle"
#     print_every = 20
#     eval_every = 1000
#     save_every = 10000
#     batch_size = 64
#     lr = 1e-3
#     epochs = 20
#     prepare_dataset('zooniverse', ml_1, output_data)
#
#     # Initialize seed
#     np.random.seed(42)
#     torch.manual_seed(42)
#
#     # Load preprocess data
#     with open(output_data, 'rb') as f:
#         dataset = pickle.load(f)
#         user_size, item_size = dataset['user_size'], dataset['item_size']
#         train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
#         train_pair = dataset['train_pair']
#     print('Load complete')
#
#     # Create dataset, model, optimizer
#     dataset = TripletUniformPair(item_size, train_user_list, train_pair, True, epochs)
#     loader = DataLoader(dataset, batch_size=batch_size, num_workers=16)
#     # model = BPR(user_size, item_size, 30, 0.025).cuda()
#     model = BPR(user_size, item_size, 100, 0.025)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     writer = SummaryWriter()
#
#     # Training
#     smooth_loss = 0
#     idx = 0
#     for u, i, j in loader:
#         optimizer.zero_grad()
#         loss = model(u, i, j)
#         loss.backward()
#         optimizer.step()
#         writer.add_scalar('train/loss', loss, idx)
#         smooth_loss = smooth_loss * 0.99 + loss * 0.01
#         if idx % print_every == (print_every - 1):
#             print('loss: %.4f' % smooth_loss)
#         if idx % eval_every == (eval_every - 1):
#             plist, rlist = precision_and_recall_k(model.W.detach(),
#                                                   model.H.detach(),
#                                                   train_user_list,
#                                                   test_user_list,
#                                                   klist=[1, 5, 10])
#             print('P@1: %.4f, P@5: %.4f P@10: %.4f, R@1: %.4f, R@5: %.4f, R@10: %.4f' % (
#             plist[0], plist[1], plist[2], rlist[0], rlist[1], rlist[2]))
#             writer.add_scalars('eval', {'P@1': plist[0],
#                                         'P@5': plist[1],
#                                         'P@10': plist[2]}, idx)
#             writer.add_scalars('eval', {'R@1': rlist[0],
#                                         'R@5': rlist[1],
#                                         'R@10': rlist[2]}, idx)
#         # if idx % save_every == (save_every - 1):
#         #     dirname = os.path.dirname(os.path.abspath(model))
#         #     os.makedirs(dirname, exist_ok=True)
#         #     torch.save(model.state_dict(), model)
#         idx += 1
