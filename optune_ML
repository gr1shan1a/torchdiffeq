import sys
import subprocess
import signal
import argparse
import random
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from modules import RunningAverageMeter, ODEJumpFunc
from utils import poisson_lmbda, exponential_hawkes_lmbda, powerlaw_hawkes_lmbda, self_inhibiting_lmbda, forward_pass, visualize, create_outpath, read_timeseries

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='none')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--dataset', type=str, default='poisson')
parser.add_argument('--suffix', type=str, default='')

parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

outpath = create_outpath(args.dataset)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')


def objective(trial):
    
    dim_c, dim_h, dim_N, dt, tspan = 5, 5, 1, 0.05, (0.0, 100.0)
    dim_hidden = trial.suggest_int('dim_hidden', 10, 100)
    num_hidden = trial.suggest_int('num_hidden', 0, 5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)


    path = "./data/point_processes/"
    TSTR = read_timeseries(path + args.dataset + "_training.csv")
    TSVA = read_timeseries(path + args.dataset + "_validation.csv")
    TSTE = read_timeseries(path + args.dataset + "_testing.csv")

    if args.dataset == "poisson":
        lmbda_va_real = poisson_lmbda(tspan[0], tspan[1], dt, 1.0, TSVA)
        lmbda_te_real = poisson_lmbda(tspan[0], tspan[1], dt, 1.0, TSTE)
    elif args.dataset == "exponential_hawkes":
        lmbda_va_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSVA, args.evnt_align)
        lmbda_te_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSTE, args.evnt_align)
    elif args.dataset == "powerlaw_hawkes":
        lmbda_va_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSVA, args.evnt_align)
        lmbda_te_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSTE, args.evnt_align)
    elif args.dataset == "self_inhibiting":
        lmbda_va_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSVA, args.evnt_align)
        lmbda_te_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSTE, args.evnt_align)


    
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=dim_hidden, num_hidden=num_hidden, ortho=True, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU())
    c0 = torch.randn(dim_c, requires_grad=True)
    h0 = torch.zeros(dim_h)
    it0 = 0
    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': c0, 'lr': 1.0e-2},
                            ], lr=lr, weight_decay=weight_decay)

    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_meter = RunningAverageMeter()
    it = it0

    
    if func.jump_type == "read":
        while it < args.niters:
            optimizer.zero_grad()
            batch_id = np.random.choice(len(TSTR), args.batch_size, replace=False)
            batch = [TSTR[seqid] for seqid in batch_id]

            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, batch, args.evnt_align)
            loss_meter.update(loss.item() / len(batch))

            func.backtrace.clear()
            loss.backward()
            print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, loss.item()/len(batch), loss_meter.avg, mete), flush=True)
            optimizer.step()
            it += 1

            
            if it % args.nsave == 0:
                
                tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSVA, args.evnt_align)

                func.backtrace.clear()
                loss.backward()
                print("iter: {}, validation loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSVA), mete), flush=True)

                
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, tsave[gtid], lmbda_va_real, tsne, range(len(TSVA)), it)

                
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + args.paramw)

    # Оценка на тестовых данных после завершения обучения
    tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSTE, args.evnt_align)
    visualize(outpath, tsave, trace, lmbda, None, None, tsave[gtid], lmbda_te_real, tsne, range(len(TSTE)), it, appendix="testing")
    print("iter: {}, testing loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSTE), mete), flush=True)

    
    return loss_meter.avg


if __name__ == '__main__':
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  

    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
