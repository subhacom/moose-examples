# hhcomp.py ---
#
# Filename: hhcomp.py
# Description:
# Author: Subhasis Ray
# Maintainer:
# Created: Tue May  7 12:11:22 2013 (+0530)
# Version:
# Last-Updated: Fri Mar 28 11:58:30 2025 (+0530)
#           By: Subhasis Ray
#     Update #: 441
# URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change log:
#
#
#
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.

"""This reimplements the Hodgkin-Huxley squid giant axon model using
formula evaluation to fill up the HHGates.

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import moose
from moose import utils

EREST_ACT = -70e-3
per_ms = 1e3


def create_na_chan(
    parent='/library', name='na', vmin=-110e-3, vmax=50e-3, vdivs=3000
):
    """
    Create a Hodgkin-Huxley Na channel under `parent`.

    vmin, vmax, vdivs: voltage range and number of divisions for gate tables

    """
    na = moose.HHChannel('%s/%s' % (parent, name))
    na.Xpower = 3
    na.Ypower = 1
    m_gate = moose.element('%s/gateX' % (na.path))
    m_gate.alphaExpr = (f'1e3 * 0.1 * (25 - 1e3 * (v - ({EREST_ACT}))) /'
                        f'(exp((25 - 1e3 * (v - ({EREST_ACT}))) / 10) - 1)')
    v = np.linspace(vmin, vmax, vdivs+1)
    alpha = (1e3 * 0.1 * (25 - 1e3*(v - EREST_ACT)) /
             (np.exp((25 - 1e3 * (v - EREST_ACT)) / 10) - 1))
    m_gate.betaExpr = f'1e3 * 4 * exp(- (v - ({EREST_ACT}))/ 18e-3)'
    beta = 1e3 * 4 * np.exp(- 1e3 * (v - EREST_ACT)/ 18)
    h_gate = moose.element('%s/gateY' % (na.path))
    h_gate.alphaExpr = f'1e3 * 0.07 * exp(- 1e3 * (v - ({EREST_ACT}))/ 20)'
    h_gate.betaExpr = f'1e3 / (exp((30 - 1e3 * (v - ({EREST_ACT}))) / 10) + 1)'
    for gate in (m_gate, h_gate):
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True
        gate.fillFromExpr()

    v = np.linspace(m_gate.min, m_gate.max, m_gate.divs + 1)
    plt.subplot(211)
    plt.plot(v, m_gate.tableA / m_gate.tableB, label='minf moose')
    plt.plot(v, alpha/(alpha+beta), '--', label='minf numpy')
    plt.plot(v, h_gate.tableA / h_gate.tableB, label='hinf moose')
    plt.legend()
    plt.subplot(212)
    plt.plot(v, 1 / m_gate.tableB, label='mtau moose')
    plt.plot(v, 1/(alpha+beta), '--', label='mtau numpy')
    plt.plot(v, 1 / h_gate.tableB, label='htau moose')
    plt.legend()
    plt.show()

    na.tick = -1
    return na


def create_k_chan(
    parent='/library', name='k', vmin=-120e-3, vmax=40e-3, vdivs=3000
):
    """Create a Hodgkin-Huxley K channel under `parent`.

    vmin, vmax, vdivs: voltage range and number of divisions for gate tables

    """
    k = moose.HHChannel('%s/%s' % (parent, name))
    k.Xpower = 4
    n_gate = moose.element('%s/gateX' % (k.path))
    n_gate.alphaExpr = (f'1e3 * 0.01 * (10 - 1e3 * (v - ({EREST_ACT}))) /'
                        f'(exp((10 - 1e3 * (v - ({EREST_ACT})))/10) - 1)')
    n_gate.betaExpr = f'1e3 * 0.125 * exp(-1e3 * (v - ({EREST_ACT})) / 80)'
    n_gate.min = -100e-3
    n_gate.max = 100e-3
    n_gate.divs = 1000
    n_gate.useInterpolation = True
    n_gate.fillFromExpr()
    v = np.linspace(n_gate.min, n_gate.max, n_gate.divs + 1)
    alpha = (1e3 * 0.01 * (10 - 1e3 * (v - (EREST_ACT))) /
             (np.exp((10 - 1e3 * (v - (EREST_ACT)))/10) - 1))
    beta = 1e3 * 0.125 * np.exp(-1e3 * (v - (EREST_ACT)) / 80)
    plt.subplot(211)
    plt.plot(v, n_gate.tableA / n_gate.tableB, label='inf_n moose')
    plt.plot(v, alpha/(alpha+beta), '--', label='inf_n numpy')
    plt.legend()
    plt.subplot(212)
    plt.plot(v, 1 / n_gate.tableB, label='tau_n moose')
    plt.plot(v, 1/(alpha+beta), '--', label='tau_n numpy')
    plt.legend()
    plt.show()
    k.tick = -1
    return k


def test_channel_gates():
    """Creates prototype channels under `/library` and plots the time
    constants (tau) and activation (minf, hinf, ninf) parameters for the
    channel gates.

    Does not execute any simulation.

    """
    lib = moose.Neutral('/library')
    na_proto = create_na_chan()
    k_proto = create_k_chan()


def create_passive_comp(
    parent='/library', name='comp', diameter=30e-6, length=0.0
):
    """Creates a single compartment with squid axon Em, Cm and Rm. Does
    not set Ra"""
    comp = moose.Compartment('%s/%s' % (parent, name))
    comp.Em = EREST_ACT + 10.613e-3
    comp.initVm = EREST_ACT
    if length <= 0:
        sarea = np.pi * diameter * diameter
    else:
        sarea = np.pi * diameter * length
    # specific conductance gm = 0.3 mS/cm^2
    comp.Rm = 1 / (0.3e-3 * sarea * 1e4)
    # Specific capacitance cm = 1 uF/cm^2
    comp.Cm = 1e-6 * sarea * 1e4
    return comp, sarea


def create_hhcomp(
    parent='/library', name='hhcomp', diameter=-30e-6, length=0.0
):
    """Create a compartment with Hodgkin-Huxley type ion channels (Na and
    K).

    Returns a 3-tuple: (compartment, nachannel, kchannel)

    """
    comp, sarea = create_passive_comp(parent, name, diameter, length)
    if moose.exists('/library/na'):
        moose.copy('/library/na', comp.path, 'na')
    else:
        create_na_chan(parent=comp.path)
    na = moose.element('%s/na' % (comp.path))
    # Na-conductance 120 mS/cm^2
    na.Gbar = 120e-3 * sarea * 1e4
    na.Ek = 115e-3 + EREST_ACT
    moose.connect(comp, 'channel', na, 'channel')
    if moose.exists('/library/k'):
        moose.copy('/library/k', comp.path, 'k')
    else:
        create_k_chan(parent=comp.path)
    k = moose.element('%s/k' % (comp.path))
    # K-conductance 36 mS/cm^2
    k.Gbar = 36e-3 * sarea * 1e4
    k.Ek = -12e-3 + EREST_ACT
    moose.connect(comp, 'channel', k, 'channel')
    return comp, na, k


def test_hhcomp():
    """Create and simulate a single spherical compartment with
    Hodgkin-Huxley Na and K channel.

    Plots Vm, injected current, channel conductances.

    """
    model = moose.Neutral('/model')
    data = moose.Neutral('/data')
    comp, na, k = create_hhcomp(parent=model.path)
    print((comp.Rm, comp.Cm, na.Ek, na.Gbar, k.Ek, k.Gbar))
    pg = moose.PulseGen('%s/pg' % (model.path))
    pg.firstDelay = 20e-3
    pg.firstWidth = 40e-3
    pg.firstLevel = 1e-9
    pg.secondDelay = 1e9
    moose.connect(pg, 'output', comp, 'injectMsg')
    inj = moose.Table('%s/pulse' % (data.path))
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')
    vm = moose.Table('%s/Vm' % (data.path))
    moose.connect(vm, 'requestOut', comp, 'getVm')
    gK = moose.Table('%s/gK' % (data.path))
    moose.connect(gK, 'requestOut', k, 'getGk')
    gNa = moose.Table('%s/gNa' % (data.path))
    moose.connect(gNa, 'requestOut', na, 'getGk')
    simdt = 1e-6
    plotdt = 1e-4
    simtime = 100e-3
    if 1:
        moose.showmsg('/clock')
        for i in range(8):
            moose.setClock(i, simdt)
        moose.setClock(8, plotdt)
        moose.reinit()
    else:
        utils.resetSim([model.path, data.path], simdt, plotdt, simmethod='ee')
        moose.showmsg('/clock')
    moose.start(simtime)
    t = np.linspace(0, simtime, len(vm.vector))
    plt.subplot(211)
    plt.plot(t, vm.vector * 1e3, label='Vm (mV)')
    plt.plot(t, inj.vector * 1e9, label='injected (nA)')
    plt.legend()
    plt.title('Vm')
    plt.subplot(212)
    plt.title('Conductance (uS)')
    plt.plot(t, gK.vector * 1e6, label='K')
    plt.plot(t, gNa.vector * 1e6, label='Na')
    plt.legend()
    plt.show()
    plt.close()


def main():
    """A compartment with hodgkin-huxley ion channels"""
    test_channel_gates()
    test_hhcomp()


if __name__ == '__main__':
    main()
#
# hhcomp.py ends here
