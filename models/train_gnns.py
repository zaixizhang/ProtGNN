import os
import argparse
import torch
import torch.nn.functional as F
import shutil
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from my_mcts import mcts
from tqdm import tqdm


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info):
    f = open('./log/hyper_search', 'a')
    f.write(info)
    f.write('\n')
    f.close()


def concrete_sample(log_alpha, beta=1.0, training=True):
    """ Sample from the instantiation of concrete distribution when training
    \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
    """
    if training:
        random_noise = torch.rand(log_alpha.shape)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
        gate_inputs = gate_inputs.sigmoid()
    else:
        gate_inputs = log_alpha.sigmoid()

    return gate_inputs


def edge_mask(inputs, training=None):
    x, embed, edge_index, prot, tmp = inputs
    nodesize = embed.shape[0]
    feature_dim = embed.shape[1]
    f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
    f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)
    f3 = prot.unsqueeze(0).repeat(nodesize*nodesize, 1)
    # using the node embedding to calculate the edge weight
    f12self = torch.cat([f1, f2, f3], dim=-1)
    h = f12self.to(model_args.device)
    for elayer in elayers:
        h = elayer(h)
    values = h.reshape(-1)
    values = concrete_sample(values, beta=tmp, training=training)
    mask_sigmoid = values.reshape(nodesize, nodesize)

    sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
    edge_mask = sym_mask[edge_index[0], edge_index[1]]

    return edge_mask


def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask


def prototype_subgraph_similarity(x, prototype):
    distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
    similarity = torch.log((distance + 1) / (distance + 1e-4))
    return distance, similarity


elayers = nn.ModuleList()
elayers.append(nn.Sequential(nn.Linear(128 * 3, 64), nn.ReLU()))
elayers.append(nn.Sequential(nn.Linear(64, 8), nn.ReLU()))
elayers.append(nn.Linear(8, 1))
elayers.to(model_args.device)


# train for graph classification
def train_GC(clst, sep):
    # attention the multi-task here
    print(clst)
    print(sep)
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio)

    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    #checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    #gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    early_stop_count = 0
    data_indices = dataloader['train'].dataset.indices
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []
        # Prototype projection
        if epoch >= train_args.proj_epochs and epoch % 10 == 0:
            gnnNets.eval()
            for i in range(output_dim * model_args.num_prototypes_per_class):
                count = 0
                best_similarity = 0
                label = i//model_args.num_prototypes_per_class
                for j in range(i*10, len(data_indices)):
                    data = dataset[data_indices[j]]
                    if data.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= 10:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print('Projection of prototype completed')
                        break

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)
        for batch in dataloader['train']:
            logits, probs, _, _, min_distances = gnnNets(batch)
            loss = criterion(logits, batch.y)
            #cluster loss
            prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y].bool()).to(model_args.device)
            cluster_cost = torch.mean(torch.min(min_distances[prototypes_of_correct_class].reshape(-1, model_args.num_prototypes_per_class), dim=1)[0])

            #seperation loss
            separation_cost = -torch.mean(torch.min(min_distances[~prototypes_of_correct_class].reshape(-1, (output_dim-1)*model_args.num_prototypes_per_class), dim=1)[0])

            #sparsity loss
            l1_mask = 1 - torch.t(gnnNets.model.prototype_class_identity).to(model_args.device)
            l1 = (gnnNets.model.last_layer.weight * l1_mask).norm(p=1)

            #diversity loss
            ld = 0
            for k in range(output_dim):
                p = gnnNets.model.prototype_vectors[k*model_args.num_prototypes_per_class: (k+1)*model_args.num_prototypes_per_class]
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device)
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            loss = loss + clst*cluster_cost + sep*separation_cost + 5e-4 * l1 + 0.00 * ld

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # report train msg
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()))
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
              f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")

        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record("Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)

    print(f"The best validation accuracy is {best_acc}.")
    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record("loss: {:.3f}, acc: {:.3f}".format(test_state['loss'], test_state['acc']))



def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state


def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean())}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def predict_GC(test_dataloader, gnnNets):
    """
    return: pred_probs --  np.array : the probability of the graph class
            predictions -- np.array : the prediction class for each graph
    """
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)

            ## record
            _, prediction = torch.max(logits, -1)
            predictions.append(prediction)
            pred_probs.append(probs)

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return pred_probs, predictions


# train for node classification task
def train_NC():
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', f"{data_args.dataset_name}")):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

    data = dataset[0]
    gnnNets_NC = GnnNets_NC(input_dim, output_dim, model_args)
    gnnNets_NC.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets_NC.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    best_val_loss = float('inf')
    best_acc = 0
    val_loss_history = []
    early_stop_count = 0
    for epoch in range(1, train_args.max_epochs + 1):
        gnnNets_NC.train()
        logits, prob, _, min_distances = gnnNets_NC(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        # cluster loss
        prototypes_of_correct_class = torch.t(gnnNets_NC.model.prototype_class_identity[:, data.y]).to(model_args.device)
        cluster_cost = torch.mean(torch.min(min_distances * prototypes_of_correct_class, dim=1)[0])

        # seperation loss
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        separation_cost = -torch.mean(torch.min(min_distances * prototypes_of_wrong_class, dim=1)[0])

        # sparsity loss
        l1_mask = 1 - torch.t(gnnNets_NC.model.prototype_class_identity).to(model_args.device)
        l1 = (gnnNets_NC.model.last_layer.weight * l1_mask).norm(p=1)

        # diversity loss
        ld = 0
        for k in range(output_dim):
            p = gnnNets_NC.model.prototype_vectors[
                k * model_args.num_prototypes_per_class: (k + 1) * model_args.num_prototypes_per_class]
            p = F.normalize(p, p=2, dim=1)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3
            matrix2 = torch.zeros(matrix1.shape).to(model_args.device)
            ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

        loss = loss + 0.1 * cluster_cost + 0.1 * separation_cost + 0.001 * ld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_info = evaluate_NC(data, gnnNets_NC, criterion)
        eval_info['epoch'] = epoch

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            val_acc = eval_info['val_acc']

        val_loss_history.append(eval_info['val_loss'])

        # only save the best model
        is_best = (eval_info['val_acc'] > best_acc)

        if eval_info['val_acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_info['val_acc']
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets_NC, model_args.model_name, eval_info['val_acc'], is_best)
            print(f'Epoch {epoch}, Train Loss: {eval_info["train_loss"]:.4f}, '
                        f'Train Accuracy: {eval_info["train_acc"]:.3f}, '
                        f'Val Loss: {eval_info["val_loss"]:.3f}, '
                        f'Val Accuracy: {eval_info["val_acc"]:.3f}')


    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    gnnNets_NC.update_state_dict(checkpoint['net'])
    eval_info = evaluate_NC(data, gnnNets_NC, criterion)
    print(f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}')


def evaluate_NC(data, gnnNets_NC, criterion):
    eval_state = {}
    gnnNets_NC.eval()

    with torch.no_grad():
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            logits, probs, _, _ = gnnNets_NC(data)
            loss = criterion(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            ## record
            eval_state['{}_loss'.format(key)] = loss
            eval_state['{}_acc'.format(key)] = acc

    return eval_state


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ProtGNN')
    parser.add_argument('--clst', type=float, default=0.0,
                        help='cluster')
    parser.add_argument('--sep', type=float, default=0.0,
                        help='separation')
    args = parser.parse_args()
    train_GC(args.clst, args.sep)
