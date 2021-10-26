#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from .. import _prop, Graph, libcore, _get_rng, PropertyMap, \
    VertexPropertyMap, Vector_size_t, Vector_double, group_vector_property, \
    perfect_prop_hash
from .. generation import condensation_graph, random_rewire, generate_sbm, \
    solve_sbm_fugacities, generate_maxent_sbm
from .. stats import remove_parallel_edges, remove_self_loops
from .. spectral import adjacency
import random
from numpy import *
import math
import numpy
import copy
import collections
from collections import OrderedDict
import itertools
import warnings

from . util import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . libgraph_tool_inference import PartitionHist, BlockPairHist

__test__ = False

def set_test(test):
    global __test__
    __test__ = test

def _bm_test():
    global __test__
    return __test__

def get_block_graph(g, B, b, vcount=None, ecount=None, rec=None, drec=None):
    if isinstance(ecount, libinference.unity_eprop_t):
        ecount = None
    if isinstance(vcount, libinference.unity_vprop_t):
        vcount = None
    avprops = []
    if vcount is not None:
        avprops.append(vcount)
    aeprops = []
    if ecount is not None:
        aeprops.append(ecount)
    if rec is not None:
        for r in rec:
            aeprops.append(r)
    if drec is not None:
        for r in drec:
            aeprops.append(r)
    cg, br, vc, ec, av, ae = condensation_graph(g, b,
                                                avprops=avprops,
                                                aeprops=aeprops,
                                                self_loops=True)
    if vcount is not None:
        vcount = av[0]
        del av[0]
    else:
        vcount = vc
    cg.vp.count = vcount

    if ecount is not None:
        ecount = ae[0]
        del ae[0]
    else:
        ecount = ec
    cg.ep.count = ecount

    if rec is not None:
        for i in range(len(rec)):
            cg.ep["rec%d" % i] = ae[0]
            del ae[0]

    if drec is not None:
        for i in range(len(drec)):
            cg.ep["drec%d" % i] = ae[0]
            del ae[0]

    rs = numpy.setdiff1d(numpy.arange(B, dtype="int"), br.fa,
                         assume_unique=True)
    if len(rs) > 0:
        cg.add_vertex(len(rs))
        br.fa[-len(rs):] = rs
    cg = Graph(cg, vorder=br)

    cg.gp.rec = cg.new_gp("object", [])
    if rec is not None:
        for i in range(len(rec)):
            cg.gp.rec.append(cg.ep["rec%d" % i])
            del cg.ep["rec%d" % i]

    cg.gp.drec = cg.new_gp("object", [])
    if drec is not None:
        for i in range(len(drec)):
            cg.gp.drec.append(cg.ep["drec%d" % i])
            del cg.ep["drec%d" % i]

    return cg

def get_entropy_args(kargs, ignore=None):
    kargs = kargs.copy()
    if ignore is not None:
        for a in ignore:
            kargs.pop(a, None)
    args = DictState(kargs)
    deg_dl_kind = args.degree_dl_kind
    del kargs["degree_dl_kind"]
    if deg_dl_kind == "entropy":
        kind = libinference.deg_dl_kind.ent
    elif deg_dl_kind == "uniform":
        kind = libinference.deg_dl_kind.uniform
    elif deg_dl_kind == "distributed":
        kind = libinference.deg_dl_kind.dist
    ea = libinference.entropy_args()
    ea.exact = args.exact
    ea.dense = args.dense
    ea.multigraph = args.multigraph
    ea.adjacency = args.adjacency
    ea.deg_entropy = args.deg_entropy
    ea.recs = args.recs
    del kargs["exact"]
    del kargs["dense"]
    del kargs["multigraph"]
    del kargs["adjacency"]
    del kargs["deg_entropy"]
    del kargs["recs"]
    if args.dl:
        ea.partition_dl = args.partition_dl
        ea.degree_dl = args.degree_dl
        ea.edges_dl = args.edges_dl
        ea.recs_dl = args.recs_dl
    else:
        ea.partition_dl = False
        ea.degree_dl = False
        ea.edges_dl = False
        ea.recs_dl = False
    ea.degree_dl_kind = kind
    ea.beta_dl = args.beta_dl
    ea.Bfield = args.Bfield
    del kargs["dl"]
    del kargs["partition_dl"]
    del kargs["degree_dl"]
    del kargs["edges_dl"]
    del kargs["recs_dl"]
    del kargs["beta_dl"]
    del kargs["Bfield"]
    kargs.pop("callback", None)
    if len(kargs) > 0:
        raise ValueError("unrecognized entropy arguments: " +
                         str(list(kargs.keys())))
    return ea

_q_cache_max_n = 10000
def init_q_cache(max_n=None):
    if max_n is None:
        max_n = _q_cache_max_n
    libinference.init_q_cache(min(_q_cache_max_n, max_n))

class BlockState(object):
    r"""The stochastic block model state of a given graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be modelled.
    b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial block labels on the vertices. If not supplied, it will be
        randomly sampled.
    B : ``int`` (optional, default: ``None``)
        Number of blocks (or vertex groups). If not supplied it will be obtained
        from the parameter ``b``.
    eweight : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
        Edge multiplicities (for multigraphs or block graphs).
    vweight : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex multiplicities (for block graphs).
    recs : list of :class:`~graph_tool.EdgePropertyMap` instances (optional, default: ``[]``)
        List of real or discrete-valued edge covariates.
    rec_types : list of edge covariate types (optional, default: ``[]``)
        List of types of edge covariates. The possible types are:
        ``"real-exponential"``, ``"real-normal"``, ``"discrete-geometric"``,
        ``"discrete-poisson"`` or ``"discrete-binomial"``.
    rec_params : list of ``dict`` (optional, default: ``[]``)
        Model hyperparameters for edge covariates. This should be a list of
        ``dict`` instances, or the string `"microcanonical"` (the default if
        nothing is specified). The keys depend on the type of edge covariate:

        ``"real-exponential"`` or ``"discrete-poisson"``
            The parameter list is ``["r", "theta"]``, corresponding to the
            parameters of the `Gamma
            <https://en.wikipedia.org/wiki/Gamma_distribution>`_ prior
            distribution.  If unspecified, the default is the "empirical Bayes"
            choice: ``r = 1.0`` and ``theta`` is the global average of the edge
            covariate.

        ``"discrete-geometric"``
            The parameter list is ``["alpha", "beta"]``, corresponding to the
            parameters of the `Beta
            <https://en.wikipedia.org/wiki/Beta_distribution>`_ prior
            distribution. If unspecified, the default is the noninformative
            choice: ``alpha = beta = 1.0``

        ``"discrete-binomial"``
            The parameter list is ``["N", "alpha", "beta"]``, corresponding to
            the number of trials ``N`` and the parameters of the `Beta
            <https://en.wikipedia.org/wiki/Beta_distribution>`_ prior
            distribution. If unspecified, the default is the noninformative
            choice, ``alpha = beta = 1.0``, and ``N`` is taken to be the maximum
            edge covarite value.

        ``"real-normal"``
            The parameter list is ``["m0", "k0", "v0", "nu0"]`` corresponding to
            the `normal
            <https://en.wikipedia.org/wiki/Normal_distribution>`_-`inverse-chi-squared
            <https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution>`_
            prior. If unspecified, the defaults are: ``m0 = rec.fa.mean()``,
            ``k0 = 1``, ``v0 = rec.fa.std() ** 2``, and ``nu0 = 3``, where
            ``rec`` is the corresponding edge covariate property map.

    clabel : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Constraint labels on the vertices. If supplied, vertices with different
        label values will not be clustered in the same group.
    pclabel : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Partition constraint labels on the vertices. This has the same
        interpretation as ``clabel``, but will be used to compute the partition
        description length.
    bfield : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Local field acting as a prior for the node partition. This should be a
        vector property map of type ``vector<double>``, and contain the
        log-probability for each node to be placed in each group.
    deg_corr : ``bool`` (optional, default: ``True``)
        If ``True``, the degree-corrected version of the blockmodel ensemble will
        be assumed, otherwise the traditional variant will be used.
    dense_bg : ``bool`` (optional, default: ``False``)
        If ``True`` a dense matrix is used for the block graph, otherwise a
        sparse matrix will be used.
    """

    def __init__(self, g, b=None, B=None, eweight=None, vweight=None, recs=[],
                 rec_types=[], rec_params=[], clabel=None, pclabel=None,
                 bfield=None, Bfield=None, deg_corr=True, dense_bg=False, **kwargs):
        kwargs = kwargs.copy()

        # initialize weights to unity, if necessary
        if eweight == "unity":
            eweight = g.new_ep("int", 1)
        elif eweight is None or isinstance(eweight, libinference.unity_eprop_t):
            eweight = libinference.unity_eprop_t()
        elif eweight.value_type() != "int32_t":
            eweight = g.own_property(eweight.copy(value_type="int32_t"))
        else:
            eweight = g.own_property(eweight)
        if vweight == "unity":
            vweight = g.new_vp("int", 1)
        elif vweight is None or isinstance(vweight, libinference.unity_vprop_t):
            vweight = libinference.unity_vprop_t()
        elif vweight.value_type() != "int32_t":
            vweight = g.own_property(vweight.copy(value_type="int32_t"))
        else:
            vweight = g.own_property(vweight)
        self.eweight = eweight
        self.vweight = vweight

        is_edge_weighted = not isinstance(eweight, libinference.unity_eprop_t)
        is_vertex_weighted = not isinstance(vweight, libinference.unity_vprop_t)
        if not is_edge_weighted and is_vertex_weighted:
            self.eweight = g.new_ep("int", 1)
        if not is_vertex_weighted and is_edge_weighted:
            self.vweight = g.new_vp("int", 1)
        self.is_weighted = is_edge_weighted or is_vertex_weighted

        # configure the main graph and block model parameters
        self.g = g

        self.deg_corr = deg_corr
        self.overlap = False
        self.degs = kwargs.pop("degs", None)
        if self.degs is None:
            if not self.is_weighted:
                self.degs = libinference.simple_degs_t()
            else:
                self.degs = "weighted"
        if self.degs == "weighted":
            if self.deg_corr:
                idx_ = self.g.vertex_index.copy("int")
                self.degs = libinference.get_block_degs(self.g._Graph__graph,
                                                        _prop("v", self.g, idx_),
                                                        self.eweight._get_any(),
                                                        self.g.num_vertices(True))
            else:
                self.degs = libinference.get_empty_degs(self.g._Graph__graph)

        if B is None and b is None:
            B = 1

        # ensure we have at most as many blocks as nodes
        if B is not None and b is None:
            B = min(B, self.g.num_vertices())

        if b is None:
            # create a random partition into B blocks.
            if B is None:
                raise ValueError("either 'b' or 'B' must be specified")
            B = min(B, self.g.num_vertices())
            ba = random.randint(0, B, self.g.num_vertices())
            ba[:B] = arange(B)        # avoid empty blocks
            if B < self.g.num_vertices():
                random.shuffle(ba)
            b = g.new_vp("int")
            b.fa = ba
            self.b = b
        else:
            # if a partition is available, we will incorporate it.
            if isinstance(b, numpy.ndarray):
                self.b = g.new_vp("int")
                self.b.fa = b
            else:
                self.b = b = g.own_property(b.copy(value_type="int32_t"))
            if B is None:
                B = int(self.b.fa.max()) + 1

        self.B = B = max(B, self.b.fa.max() + 1)

        self.rec = [self.g.own_property(p) for p in recs]
        for i in range(len(self.rec)):
            if self.rec[i].value_type() != "double":
                self.rec[i] = self.rec[i].copy("double")
        self.drec = kwargs.pop("drec", None)
        if self.drec is None:
            self.drec = []
            for rec in self.rec:
                self.drec.append(self.g.new_ep("double", rec.fa ** 2))
        else:
            if (isinstance(self.drec, PropertyMap) and
                self.drec.value_type() == "vector<double>"):
                self.drec = ungroup_vector_property(self.drec,
                                                    range(len(self.drec[next(g.edges())])))
            self.drec = [self.g.own_property(p) for p in self.drec]

        rec_types = list(rec_types)
        rec_params = list(rec_params)

        # if len(rec_params) < len(rec_types):
        #     rec_params += [{} for i in range((len(rec_types) -
        #                                       len(rec_params)))]

        if len(self.rec) > 0 and rec_types[0] != libinference.rec_type.count:
            rec_types.insert(0, libinference.rec_type.count)
            rec_params.insert(0, {})
            if isinstance(self.eweight, libinference.unity_eprop_t):
                self.rec.insert(0, self.g.new_ep("double", 1))
            else:
                self.rec.insert(0, self.eweight.copy("double"))
            self.drec.insert(0, self.g.new_ep("double"))

        # Construct block-graph
        self.bg = get_block_graph(g, B, self.b, self.vweight, self.eweight,
                                  rec=self.rec, drec=self.drec)
        self.bg.set_fast_edge_removal()

        self.mrs = self.bg.ep["count"]
        self.wr = self.bg.vp["count"]

        self.mrp = self.bg.degree_property_map("out", weight=self.mrs)

        if g.is_directed():
            self.mrm = self.bg.degree_property_map("in", weight=self.mrs)
        else:
            self.mrm = self.mrp

        if pclabel is not None:
            if isinstance(pclabel, PropertyMap):
                self.pclabel = self.g.own_property(pclabel).copy("int")
            else:
                self.pclabel = self.g.new_vp("int")
                self.pclabel.fa = pclabel
        else:
            self.pclabel = self.g.new_vp("int")

        if clabel is not None:
            if isinstance(clabel, PropertyMap):
                self.clabel = self.g.own_property(clabel).copy("int")
            else:
                self.clabel = self.g.new_vp("int")
                self.clabel.fa = clabel
        elif self.pclabel.fa.max() > 0:
            self.clabel = self.pclabel
        else:
            self.clabel = self.g.new_vp("int")

        self._coupled_state = None

        if not self._check_clabel():
            raise ValueError("provided clabel is inconsistent with node partition")
        if not self._check_clabel(clabel=self.pclabel):
            raise ValueError("provided pclabel is inconsistent with node partition")
        if not self._check_clabel(b=self.pclabel, clabel=self.clabel):
            raise ValueError("provided pclabel and clabel are inconsistent")

        self.bclabel = self.get_bclabel()
        self.hclabel = self.bg.new_vp("int")

        self.dense_bg = dense_bg
        self.use_hash = not self.dense_bg
        self.use_rmap = kwargs.pop("use_rmap", False)

        self.merge_map = kwargs.pop("merge_map",
                                    self.g.vertex_index.copy("int"))
        self.merge_map = self.g.own_property(self.merge_map)

        self.candidate_blocks = Vector_size_t()
        self.candidate_pos = self.bg.new_vp("int")
        self.empty_blocks = Vector_size_t()
        self.empty_pos = self.bg.new_vp("int")

        self._init_recs(self.rec, rec_types, rec_params)
        self.recdx = libcore.Vector_double(len(self.rec))
        self.Lrecdx = kwargs.pop("Lrecdx", None)
        if self.Lrecdx is None:
            self.Lrecdx = libcore.Vector_double(len(self.rec)+1)
            self.Lrecdx[0] = -1
        self.Lrecdx.resize(len(self.rec)+1)
        self.epsilon = kwargs.pop("epsilon", None)
        if self.epsilon is None:
            self.epsilon = libcore.Vector_double(len(self.rec))
            for i in range(len(self.rec)):
                idx = self.rec[i].a != 0
                if not isinstance(self.eweight, libinference.unity_eprop_t):
                    idx = numpy.logical_and(idx, self.eweight.a > 0)
                if numpy.any(idx):
                    self.epsilon[i] = abs(self.rec[i].a[idx]).min() / 10

        self.bfield = g.new_vp("vector<double>") if bfield is None else self.g.own_property(bfield)
        if self.bfield.value_type() != "vector<double>":
            raise ValueError("'bfield' property map must be of type 'vector<double>'")
        if Bfield is None:
            self.Bfield = Vector_double()
        else:
            self.Bfield = Vector_double(len(Bfield))
            self.Bfield.a = Bfield
        self._abg = self.bg._get_any()
        self._avweight = self.vweight._get_any()
        self._aeweight = self.eweight._get_any()
        self._adegs = self.degs._get_any()
        self._state = libinference.make_block_state(self, _get_rng())
        self.bg.properties.clear()

        assert all(self.recdx.a >= 0), self.recdx.a

        if deg_corr:
            init_q_cache(max(2 * max(self.get_E(), self.get_N()), 100))

        self._entropy_args = dict(adjacency=True, deg_entropy=True, dl=True,
                                  partition_dl=True, degree_dl=True,
                                  degree_dl_kind="distributed", edges_dl=True,
                                  dense=False, multigraph=True, exact=True,
                                  recs=True, recs_dl=True, beta_dl=1.,
                                  Bfield=True)

        if len(kwargs) > 0:
            warnings.warn("unrecognized keyword arguments: " +
                          str(list(kwargs.keys())))

    def _init_recs(self, recs, rec_types, rec_params):
        if len(rec_types) != len(recs):
            raise ValueError("The size of 'rec_types' (%d) must be the same of 'recs' (%d)" %
                             (len(rec_types), len(recs)))
        self.rec_types = libcore.Vector_int32_t()
        for rec_type in rec_types:
            if rec_type == "real-exponential":
                rt = libinference.rec_type.real_exponential
            elif rec_type == "real-normal":
                rt = libinference.rec_type.real_normal
            elif rec_type == "discrete-geometric":
                rt = libinference.rec_type.discrete_geometric
            elif rec_type == "discrete-poisson":
                rt = libinference.rec_type.discrete_poisson
            elif rec_type == "discrete-binomial":
                rt = libinference.rec_type.discrete_binomial
            else:
                rt = rec_type
            self.rec_types.append(rt)

        self.brec = [self.bg.own_property(p) for p in self.bg.gp.rec]
        self.bdrec = [self.bg.own_property(p) for p in self.bg.gp.drec]
        self.brecsum = self.bg.new_vp("double")

        self.rec_params = rec_params = list(rec_params)
        while len(rec_params) < len(self.rec_types):
            rec_params.append("microcanonical")
        self.wparams = libcore.Vector_Vector_double()

        for i, rt in enumerate(self.rec_types):
            ps = Vector_double()
            if rt == libinference.rec_type.count:
                defaults = OrderedDict()
            elif rt in [libinference.rec_type.real_exponential,
                        libinference.rec_type.discrete_poisson]:
                if rec_params[i] != "microcanonical":
                    defaults = OrderedDict([("alpha", 1),
                                            ("beta", self.rec[i].fa.mean())])
                else:
                    defaults = OrderedDict([("alpha", numpy.nan),
                                            ("beta", numpy.nan)])
            elif rt == libinference.rec_type.real_normal:
                if rec_params[i] != "microcanonical":
                    defaults = OrderedDict([("m0", self.rec[i].fa.mean()),
                                            ("k0", 1),
                                            ("v0", self.rec[i].fa.std() ** 2),
                                            ("nu0", 3)])
                else:
                    defaults = OrderedDict([("m0", numpy.nan),
                                            ("k0", numpy.nan),
                                            ("v0", numpy.nan),
                                            ("nu0", numpy.nan)])
            elif rt == libinference.rec_type.discrete_geometric:
                if rec_params[i] != "microcanonical":
                    defaults = OrderedDict([("alpha", 1),
                                            ("beta", 1)])
                else:
                    defaults = OrderedDict([("alpha", numpy.nan),
                                            ("beta", numpy.nan)])
            elif rt == libinference.rec_type.discrete_binomial:
                if rec_params[i] != "microcanonical":
                    defaults = OrderedDict([("N", self.rec[i].fa.max()),
                                            ("alpha", 1),
                                            ("beta", 1)])
                else:
                    defaults = OrderedDict([("N", self.rec[i].fa.max()),
                                            ("alpha", numpy.nan),
                                            ("beta", numpy.nan)])

            ks = list(defaults.keys())
            if rec_params[i] != "microcanonical":
                defaults.update(rec_params[i])
            rec_params[i] = defaults.copy()
            for k in ks:
                ps.append(defaults.pop(k))
            if len(defaults) > 0:
                raise ValueError("unknown parameters for weight type: " +
                                 str(list(defaults.keys())))
            self.wparams.append(ps)

    def get_rec_params(self):
        """Get model hyperparameters for edge covariates."""
        params = []
        for rt, ps in zip(self.rec_types, self.rec_params):
            if rt == libinference.rec_type.count:
                continue
            p = dict(ps)
            params.append(p)
        return params

    def set_rec_params(self, params):
        """Update model hyperparameters for edge covariates."""
        for ps, ws, nps in zip(self.rec_params[1:], self.wparams[1:], params):
            ps.update(nps)
            for i, (k, v) in enumerate(ps.items()):
                ws[i] = v

    def __repr__(self):
        return "<BlockState object with %d blocks (%d nonempty),%s%s for graph %s, at 0x%x>" % \
            (self.B, self.get_nonempty_B(),
             " degree-corrected," if self.deg_corr else "",
             ((" with %d edge covariate%s," % (len(self.rec_types) - 1,
                                               "s" if len(self.rec_types) > 2 else ""))
              if len(self.rec_types) > 0 else ""),
             str(self.g), id(self))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        g = copy.deepcopy(self.g, memo)
        return self.copy(g=g)

    def copy(self, g=None, eweight=None, vweight=None, b=None, B=None,
             deg_corr=None, clabel=None, overlap=False, pclabel=None,
             bfield=None, dense_bg=None, **kwargs):
        r"""Copies the block state. The parameters override the state properties, and
         have the same meaning as in the constructor."""

        if not overlap:
            state = BlockState(self.g if g is None else g,
                               eweight=self.eweight if eweight is None else eweight,
                               vweight=self.vweight if vweight is None else vweight,
                               b=self.b.copy() if b is None else b,
                               B=(self.B if b is None else None) if B is None else B,
                               clabel=self.clabel if clabel is None else clabel,
                               pclabel=self.pclabel if pclabel is None else pclabel,
                               bfield=self.bfield if bfield is None else bfield,
                               deg_corr=self.deg_corr if deg_corr is None else deg_corr,
                               dense_bg=self.dense_bg if dense_bg is None else dense_bg,
                               degs=self.degs.copy() if eweight is None else None,
                               merge_map=kwargs.pop("merge_map",
                                                    self.merge_map.copy()),
                               recs=kwargs.pop("recs", self.rec),
                               drec=kwargs.pop("drec", self.drec),
                               rec_types=kwargs.pop("rec_types", self.rec_types),
                               rec_params=kwargs.pop("rec_params",
                                                     self.rec_params),
                               Lrecdx=kwargs.pop("Lrecdx", self.Lrecdx.copy()),
                               epsilon=kwargs.pop("epsilon",self.epsilon.copy()),
                               **kwargs)
        else:
            state = OverlapBlockState(self.g if g is None else g,
                                      b=self.b.copy() if b is None else b,
                                      B=(self.B if b is None else None) if B is None else B,
                                      recs=kwargs.pop("recs", self.rec),
                                      drec=kwargs.pop("drec", self.drec),
                                      rec_types=kwargs.pop("rec_types",
                                                           self.rec_types),
                                      rec_params=kwargs.pop("rec_params",
                                                            self.rec_params),
                                      clabel=self.clabel if clabel is None else clabel,
                                      pclabel=self.pclabel if pclabel is None else pclabel,
                                      deg_corr=self.deg_corr if deg_corr is None else deg_corr,
                                      dense_bg=self.dense_bg if dense_bg is None else dense_bg,
                                      Lrecdx=kwargs.pop("Lrecdx", self.Lrecdx.copy()),
                                      epsilon=kwargs.pop("epsilon",self.epsilon.copy()),
                                      **kwargs)

        if self._coupled_state is not None:
            state._couple_state(state.get_block_state(b=state.get_bclabel(),
                                                      copy_bg=False,
                                                      vweight="nonempty",
                                                      Lrecdx=state.Lrecdx),
                                self._coupled_state[1])
        return state


    def __getstate__(self):
        state = dict(g=self.g,
                     eweight=self.eweight if self.is_weighted else None,
                     vweight=self.vweight if self.is_weighted else None,
                     b=self.b,
                     B=self.B,
                     clabel=self.clabel,
                     pclabel=self.pclabel,
                     bfield=self.bfield,
                     deg_corr=self.deg_corr,
                     dense_bg=self.dense_bg,
                     recs=self.rec,
                     drec=self.drec,
                     rec_types=self.rec_types,
                     rec_params=self.rec_params,
                     merge_map=self.merge_map)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def get_block_state(self, b=None, vweight=False, **kwargs):
        r"""Returns a :class:`~graph_tool.inference.blockmodel.BlockState` corresponding
        to the block graph (i.e. the blocks of the current state become the
        nodes). The parameters have the same meaning as the in the
        constructor. If ``vweight == True`` the nodes of the block state are
        weighted with the node counts.

        """

        deg_corr = kwargs.pop("deg_corr", self.deg_corr if vweight == True else False)
        copy_bg = kwargs.pop("copy_bg", True)

        if vweight and deg_corr:
            if isinstance(self.degs, libinference.simple_degs_t):
                degs = libinference.get_block_degs(self.g._Graph__graph,
                                                   _prop("v", self.g, self.b),
                                                   self.eweight._get_any(),
                                                   self.get_B())
            else:
                degs = libinference.get_weighted_block_degs(self.g._Graph__graph,
                                                            self.degs,
                                                            _prop("v", self.g,
                                                                  self.b),
                                                            self.get_B())
        else:
            degs = None

        if copy_bg:
            self.bg.ep["eweight"] = self.mrs
            for i in range(len(self.brec)):
                self.bg.ep["brec%d"%i] = self.brec[i]
                self.bg.ep["bdrec%d"%i] = self.bdrec[i]
            bg = self.bg.copy()
            eweight = bg.ep["eweight"]
            brec = [bg.ep["brec%d"%i] for i in range(len(self.brec))]
            bdrec = [bg.ep["bdrec%d"%i] for i in range(len(self.bdrec))]
            bg.properties.clear()
        else:
            bg = self.bg
            eweight = self.mrs
            brec = self.brec
            bdrec = self.bdrec
            # if self.g.get_vertex_filter()[0] is not None:
            #     bg = GraphView(bg, vfilt=numpy.ones(bg.num_vertices()))

        copy_coupled = False
        recs = False
        if vweight == "nonempty":
            vweight = bg.new_vp("int", self.wr.a > 0)
        elif vweight == "unity":
            vweight = bg.new_vp("int", 1)
        elif vweight == True:
            if copy_bg:
                vweight = bg.own_property(self.wr.copy())
            else:
                vweight = self.wr
            recs = True
            copy_coupled = True
            kwargs["Lrecdx"] = kwargs.get("Lrecdx", self.Lrecdx.copy())
        else:
            vweight = None

        if recs:
            rec_types = kwargs.pop("rec_types", self.rec_types)
            recs = kwargs.pop("recs", brec)
            drec = kwargs.pop("drec", bdrec)
            rec_params = kwargs.pop("rec_params", self.rec_params)
        else:
            recs = []
            drec = None
            rec_types = []
            rec_params = []
            for rt, rp, r in zip(self.rec_types, self.wparams, brec):
                if rt == libinference.rec_type.count:
                    recs.append(bg.new_ep("double", eweight.fa > 0))
                    rec_types.append(rt)
                    rec_params.append("microcanonical")
                elif numpy.isnan(rp.a).sum() == 0:
                    continue
                elif rt in [libinference.rec_type.discrete_geometric,
                            libinference.rec_type.discrete_binomial,
                            libinference.rec_type.discrete_poisson]:
                    recs.append(r)
                    rec_types.append(libinference.rec_type.discrete_geometric)
                    rec_params.append("microcanonical")
                elif rt == libinference.rec_type.real_exponential:
                    recs.append(r)
                    rec_types.append(rt)
                    rec_params.append("microcanonical")
                elif rt == libinference.rec_type.real_normal:
                    recs.append(r)
                    rec_types.append(rt)
                    rec_params.append("microcanonical")
            rec_params = kwargs.pop("rec_params", rec_params)

        state = BlockState(bg,
                           eweight=eweight,
                           vweight=vweight,
                           b=bg.vertex_index.copy("int") if b is None else b,
                           deg_corr=deg_corr,
                           degs=degs,
                           rec_types=rec_types,
                           recs=recs,
                           drec=drec,
                           rec_params=rec_params,
                           clabel=kwargs.pop("clabel", self.get_bclabel()),
                           pclabel=kwargs.pop("pclabel", self.get_bpclabel()),
                           dense_bg=self.dense_bg,
                           epsilon=kwargs.pop("epsilon",
                                              self.epsilon.copy()),
                           **kwargs)

        if copy_coupled and self._coupled_state is not None:
            state._couple_state(state.get_block_state(b=state.get_bclabel(),
                                                      copy_bg=False,
                                                      vweight="nonempty",
                                                      Lrecdx=state.Lrecdx),
                                self._coupled_state[1])
        return state

    def get_E(self):
        r"Returns the total number of edges."
        return int(self.eweight.fa.sum()) if self.is_weighted else self.g.num_edges()

    def get_N(self):
        r"Returns the total number of nodes."
        return int(self.vweight.fa.sum()) if self.is_weighted else self.g.num_vertices()

    def get_B(self):
        r"Returns the total number of blocks."
        return self.bg.num_vertices()

    def get_nonempty_B(self):
        r"Returns the total number of nonempty blocks."
        return int((self.wr.a > 0).sum())

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = numpy.array(self.wr.a, dtype="double")
        w = w[w>0]
        w /= w.sum()
        return numpy.exp(-(w*log(w)).sum())

    def get_bclabel(self, clabel=None):
        r"""Returns a :class:`~graph_tool.VertexPropertyMap` corresponding to constraint
        labels for the block graph."""

        bclabel = self.bg.new_vertex_property("int")
        reverse_map(self.b, bclabel)
        if clabel is None:
            clabel = self.clabel
        pmap(bclabel, clabel)
        return bclabel

    def _set_bclabel(self, bstate):
        self.bclabel.a = bstate.b.a
        self.clabel.a = self.b.a
        pmap(self.clabel, self.bclabel)

    def get_bpclabel(self):
        r"""Returns a :class:`~graph_tool.VertexPropertyMap` corresponding to partition
        constraint labels for the block graph."""

        return self.get_bclabel(self.pclabel)

    def _check_clabel(self, clabel=None, b=None):
        if b is None:
            b = self.b
        if clabel is None:
            clabel = self.clabel
        joint = group_vector_property([b, clabel])
        joint = perfect_prop_hash([joint])[0]
        joint = b.fa.copy()
        b = b.fa.copy()
        continuous_map(joint)
        continuous_map(b)
        if not (b == joint).all():
            return False
        # if self._coupled_state is not None:
        #     b = self.bclabel
        #     return (b.fa == self._coupled_state[0].b.fa).all()
        return True

    def _couple_state(self, state, entropy_args):
        if state is None:
            self._coupled_state = None
            self._state.decouple_state()
        else:
            if _bm_test():
                assert state.g is self.bg
            self._coupled_state = (state, entropy_args)
            eargs = get_entropy_args(dict(self._entropy_args,
                                          **entropy_args))
            self._state.couple_state(state._state, eargs)
            #self._set_bclabel(state)

    def get_blocks(self):
        r"""Returns the property map which contains the block labels for each vertex."""
        return self.b

    def get_state(self):
        """Alias to :meth:`~BlockState.get_blocks`."""
        return self.get_blocks()

    def set_state(self, b):
        r"""Sets the internal partition of the state."""
        if not isinstance(b, VertexPropertyMap):
            b = self.g.new_vp("int32_t", vals=b)
        if b.value_type() != "int32_t":
            b = b.copy("int32_t")
        self._state.set_partition(_prop("v", self.g, b))

    def get_bg(self):
        r"""Returns the block graph."""
        return self.bg

    def get_ers(self):
        r"""Returns the edge property map of the block graph which contains the
        :math:`e_{rs}` matrix entries.  For undirected graphs, the diagonal
        values (self-loops) contain :math:`e_{rr}/2`."""
        return self.mrs

    def get_er(self):
        r"""Returns the vertex property map of the block graph which contains the number
        :math:`e_r` of half-edges incident on block :math:`r`. If the graph is
        directed, a pair of property maps is returned, with the number of
        out-edges :math:`e^+_r` and in-edges :math:`e^-_r`, respectively."""
        if self.bg.is_directed():
            return self.mrp, self.mrm
        else:
            return self.mrp

    def get_nr(self):
        r"""Returns the vertex property map of the block graph which contains the block
        sizes :math:`n_r`."""
        return self.wr

    def entropy(self, adjacency=True, dl=True, partition_dl=True,
                degree_dl=True, degree_dl_kind="distributed", edges_dl=True,
                dense=False, multigraph=True, deg_entropy=True, recs=True,
                recs_dl=True, beta_dl=1., Bfield=True, exact=True, **kwargs):
        r"""Calculate the entropy (a.k.a. negative log-likelihood) associated
        with the current block partition.

        Parameters
        ----------
        adjacency : ``bool`` (optional, default: ``True``)
            If ``True``, the adjacency term of the description length will be
            included.
        dl : ``bool`` (optional, default: ``True``)
            If ``True``, the description length for the parameters will be
            included.
        partition_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the partition description length
            will be included.
        degree_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the degree sequence description
            length will be included (for degree-corrected models).
        degree_dl_kind : ``str`` (optional, default: ``"distributed"``)
            This specifies the prior used for the degree sequence. It must be
            one of: ``"uniform"``, ``"distributed"`` (default) or ``"entropy"``.
        edges_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the edge matrix description length
            will be included.
        dense : ``bool`` (optional, default: ``False``)
            If ``True``, the "dense" variant of the entropy will be computed.
        multigraph : ``bool`` (optional, default: ``True``)
            If ``True``, the multigraph entropy will be used.
        deg_entropy : ``bool`` (optional, default: ``True``)
            If ``True``, the degree entropy term that is independent of the
            network partition will be included (for degree-corrected models).
        recs : ``bool`` (optional, default: ``True``)
            If ``True``, the likelihood for real or discrete-valued edge
            covariates is computed.
        recs_dl : ``bool`` (optional, default: ``True``)
            If ``True``, and ``dl == True`` the edge covariate description
            length will be included.
        beta_dl : ``double`` (optional, default: ``1.``)
            Prior inverse temperature.
        exact : ``bool`` (optional, default: ``True``)
            If ``True``, the exact expressions will be used. Otherwise,
            Stirling's factorial approximation will be used for some terms.

        Notes
        -----

        The "entropy" of the state is the negative log-likelihood of the
        microcanonical SBM, that includes the generated graph
        :math:`\boldsymbol{A}` and the model parameters
        :math:`\boldsymbol{\theta}`,

        .. math::

           \Sigma &= - \ln P(\boldsymbol{A},\boldsymbol{\theta}) \\
                  &= - \ln P(\boldsymbol{A}|\boldsymbol{\theta}) - \ln P(\boldsymbol{\theta}).

        This value is also called the `description length
        <https://en.wikipedia.org/wiki/Minimum_description_length>`_ of the data,
        and it corresponds to the amount of information required to describe it
        (in `nats <https://en.wikipedia.org/wiki/Nat_(unit)>`_).

        For the traditional blockmodel (``deg_corr == False``), the model
        parameters are :math:`\boldsymbol{\theta} = \{\boldsymbol{e},
        \boldsymbol{b}\}`, where :math:`\boldsymbol{e}` is the matrix of edge
        counts between blocks, and :math:`\boldsymbol{b}` is the partition of the
        nodes into blocks. For the degree-corrected blockmodel (``deg_corr ==
        True``), we have an additional set of parameters, namely the degree
        sequence :math:`\boldsymbol{k}`.

        For the traditional blockmodel, the model likelihood is

        .. math::

            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b}) &= \frac{\prod_{r<s}e_{rs}!\prod_re_{rr}!!}{\prod_rn_r^{e_r}}\times \frac{1}{\prod_{i<j}A_{ij}!\prod_iA_{ii}!!},\\
            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b}) &= \frac{\prod_{rs}e_{rs}!}{\prod_rn_r^{e_r}}\times \frac{1}{\prod_{ij}A_{ij}!},

        for undirected and directed graphs, respectively, where :math:`e_{rs}`
        is the number of edges from block :math:`r` to :math:`s` (or the number
        of half-edges for the undirected case when :math:`r=s`), and :math:`n_r`
        is the number of vertices in block :math:`r` .

        For the degree-corrected variant the equivalent expressions are

        .. math::

            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b},\boldsymbol{k}) &= \frac{\prod_{r<s}e_{rs}!\prod_re_{rr}!!}{\prod_re_r!}\times \frac{\prod_ik_i!}{\prod_{i<j}A_{ij}!\prod_iA_{ii}!!},\\
            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b},\boldsymbol{k}) &= \frac{\prod_{rs}e_{rs}!}{\prod_re_r^+!\prod_re_r^-!}\times \frac{\prod_ik_i^+!\prod_ik_i^-!}{\prod_{ij}A_{ij}!},

        where :math:`e_r = \sum_se_{rs}` is the number of half-edges incident on
        block :math:`r`, and :math:`e^+_r = \sum_se_{rs}` and :math:`e^-_r =
        \sum_se_{sr}` are the numbers of out- and in-edges adjacent to block
        :math:`r`, respectively.

        If ``exact == False``, `Stirling's approximation
        <https://en.wikipedia.org/wiki/Stirling%27s_approximation>`_ is used in
        the above expression.

        If ``dense == True``, the likelihood for the non-degree-corrected model
        becomes instead

        .. math::

            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b})^{-1} &= \prod_{r<s}{n_rn_s\choose e_{rs}}\prod_r{{n_r\choose 2}\choose e_{rr}/2},\\
            P(\boldsymbol{A}|\boldsymbol{e},\boldsymbol{b})^{-1} &= \prod_{rs}{n_rn_s\choose e_{rs}}

        if ``multigraph == False``, otherwise we replace :math:`{n\choose
        m}\to\left(\!\!{n\choose m}\!\!\right)` above, where
        :math:`\left(\!\!{n\choose m}\!\!\right) = {n+m-1\choose m}`.  A "dense"
        entropy for the degree-corrected model is not available, and if
        requested will raise a :exc:`NotImplementedError`.

        If ``dl == True``, the description length :math:`\mathcal{L} = -\ln
        P(\boldsymbol{\theta})` of the model will be returned as well. The terms
        :math:`P(\boldsymbol{e})` and :math:`P(\boldsymbol{b})` are described in
        described in :func:`model_entropy`.

        For the degree-corrected model we need to specify the prior
        :math:`P(\boldsymbol{k})` for the degree sequence as well. Here there
        are three options:

        1. ``degree_dl_kind == "uniform"``

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\left(\!\!{n_r\choose e_r}\!\!\right)^{-1}.

            This corresponds to a noninformative prior, where the degrees are
            sampled from a uniform distribution.

        2. ``degree_dl_kind == "distributed"`` (default)

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) = \prod_r\frac{\prod_k\eta_k^r!}{n_r!} \prod_r q(e_r, n_r)^{-1}

            with :math:`\eta_k^r` being the number of nodes with degree
            :math:`k` in group :math:`r`, and :math:`q(n,m)` being the number of
            `partitions
            <https://en.wikipedia.org/wiki/Partition_(number_theory)>`_ of
            integer :math:`n` into at most :math:`m` parts.

            This corresponds to a prior for the degree sequence conditioned on
            the degree frequencies, which are themselves sampled from a uniform
            hyperprior. This option should be preferred in most cases.

        3. ``degree_dl_kind == "entropy"``

            .. math::

                P(\boldsymbol{k}|\boldsymbol{e},\boldsymbol{b}) \approx \prod_r\exp\left(-n_rH(\boldsymbol{k}_r)\right)

            where :math:`H(\boldsymbol{k}_r) = -\sum_kp_r(k)\ln p_r(k)` is the
            entropy of the degree distribution inside block :math:`r`.

            Note that, differently from the other two choices, this represents
            only an approximation of the description length. It is meant to be
            used only for comparison purposes, and should be avoided in practice.

        For the directed case, the above expressions are duplicated for the in-
        and out-degrees.

        References
        ----------

        .. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric
           Bayesian inference of the microcanonical stochastic block model",
           Phys. Rev. E 95 012317 (2017), :doi:`10.1103/PhysRevE.95.012317`,
           :arxiv:`1610.02703`
        .. [peixoto-hierarchical-2014] Tiago P. Peixoto, "Hierarchical block
           structures and high-resolution model selection in large networks ",
           Phys. Rev. X 4, 011047 (2014), :doi:`10.1103/PhysRevX.4.011047`,
           :arxiv:`1310.4377`.
        .. [peixoto-weighted-2017] Tiago P. Peixoto, "Nonparametric weighted
           stochastic block models", Phys. Rev. E 97, 012306 (2018),
           :doi:`10.1103/PhysRevE.97.012306`, :arxiv:`1708.01432`

        """

        eargs = get_entropy_args(locals(), ignore=["self", "kwargs"])

        if _bm_test() and kwargs.get("test", True):
            args = dict(**locals())
            args.update(**kwargs)
            del args["eargs"]
            del args["self"]
            del args["kwargs"]

        kwargs = kwargs.copy()

        S = self._state.entropy(eargs, kwargs.pop("propagate", False))
        dl_S = 0

        callback = kwargs.pop("callback", None)
        if callback is not None:
            dl_S += callback(self)

        S += beta_dl * dl_S

        #### modification for unlabelled graphs
        import scipy
        n_h = self.get_nr().a
        n = n_h.sum()
        S = (S - scipy.special.gammaln(n + 1) + scipy.special.gammaln(n_h + 1).sum()).item()
        #### modification for unlabelled graphs

        if kwargs.pop("test", True) and _bm_test():
            assert not isnan(S) and not isinf(S), \
                "invalid entropy %g (%s) " % (S, str(args))

            args["test"] = False
            state_copy = self.copy()
            Salt = state_copy.entropy(**args)

            assert math.isclose(S, Salt, abs_tol=1e-8), \
                "entropy discrepancy after copying (%g %g %g)" % (S, Salt,
                                                                  S - Salt)

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return S

    def get_matrix(self):
        r"""Returns the block matrix (as a sparse :class:`~scipy.sparse.csr_matrix`),
        which contains the number of edges between each block pair.

        .. warning::

           This corresponds to the adjacency matrix of the block graph, which by
           convention includes twice the amount of edges in the diagonal entries
           if the graph is undirected.

        Examples
        --------

        .. testsetup:: get_matrix

           gt.seed_rng(42)
           np.random.seed(42)
           from pylab import *

        .. doctest:: get_matrix

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=5, deg_corr=True)
           >>> state.mcmc_sweep(niter=1000)
           (...)
           >>> m = state.get_matrix()
           >>> figure()
           <...>
           >>> matshow(m.todense())
           <...>
           >>> savefig("bloc_mat.svg")

        .. figure:: bloc_mat.*
           :align: center

           A  5x5 block matrix.

       """

        return adjacency(self.bg, weight=self.mrs)

    def virtual_vertex_move(self, v, s, **kwargs):
        r"""Computes the entropy difference if vertex ``v`` is moved to block ``s``. The
        remaining parameters are the same as in
        :meth:`graph_tool.inference.blockmodel.BlockState.entropy`."""
        return self._state.virtual_move(int(v), self.b[v], s,
                                        get_entropy_args(dict(self._entropy_args,
                                                              **kwargs)))

    def move_vertex(self, v, s):
        r"""Move vertex ``v`` to block ``s``.

        This optionally accepts a list of vertices and blocks to move
        simultaneously.
        """
        if not isinstance(v, collections.Iterable):
            self._state.move_vertex(int(v), s)
        else:
            self._state.move_vertices(numpy.asarray(v, dtype="uint64"),
                                      numpy.asarray(s, dtype="uint64"))

    def remove_vertex(self, v):
        r"""Remove vertex ``v`` from its current group.

        This optionally accepts a list of vertices to remove.

        .. warning::

           This will leave the state in an inconsistent state before the vertex
           is returned to some other group, or if the same vertex is removed
           twice.
        """
        if isinstance(v, collections.Iterable):
            if not isinstance(v, numpy.ndarray):
                v = list(v)
            self._state.remove_vertices(numpy.asarray(v, dtype="uint64"))
        else:
            self._state.remove_vertex(int(v))

    def add_vertex(self, v, r):
        r"""Add vertex ``v`` to block ``r``.

        This optionally accepts a list of vertices and blocks to add.

        .. warning::

           This can leave the state in an inconsistent state if a vertex is
           added twice to the same group.
        """
        if isinstance(v, collections.Iterable):
            if not isinstance(v, numpy.ndarray):
                v = list(v)
            if not isinstance(r, numpy.ndarray):
                r = list(r)
            self._state.add_vertices(numpy.asarray(v, dtype="uint64"),
                                     numpy.asarray(r, dtype="uint64"))
        else:
            self._state.add_vertex(int(v), r)

    def merge_vertices(self, u, v):
        r"""Merge vertex ``u`` into ``v``.

        .. warning::

           This modifies the underlying graph.
        """
        self.move_vertex(u, self.b[v])
        self._state.merge_vertices(int(u), int(v))

    def sample_vertex_move(self, v, c=1., d=.1):
        r"""Sample block membership proposal of vertex ``v`` according to real-valued
        sampling parameters ``c`` and ``d``: For :math:`c\to 0` the blocks are sampled
        according to the local neighborhood and their connections; for
        :math:`c\to\infty` the blocks are sampled randomly. With a probability
        ``d``, a new (empty) group is sampled.
        """
        return self._state.sample_block(int(v), c, d, _get_rng())

    def get_move_prob(self, v, s, c=1., d=.1, reverse=False):
        r"""Compute the probability of a move proposal for vertex ``v`` to block ``s``
        according to sampling parameters ``c`` and ``d``, as obtained with
        :meth:`graph_tool.inference.blockmodel.BlockState.sample_vertex_move`. If ``reverse
        == True``, the reverse probability of moving the node back from block
        ``s`` to its current one is obtained.
        """
        if not reverse:
            return self._state.get_move_prob(int(v), self.b[v], s, c, d, False)
        else:
            return self._state.get_move_prob(int(v), s, self.b[v], c, d, True)

    def get_edges_prob(self, missing, spurious=[], entropy_args={}):
        r"""Compute the joint log-probability of the missing and spurious edges given by
        ``missing`` and ``spurious`` (a list of ``(source, target)``
        tuples, or :meth:`~graph_tool.Edge` instances), together with the
        observed edges.

        More precisely, the log-likelihood returned is

        .. math::

            \ln \frac{P(\boldsymbol G + \delta \boldsymbol G | \boldsymbol b)}{P(\boldsymbol G| \boldsymbol b)}

        where :math:`\boldsymbol G + \delta \boldsymbol G` is the modified graph
        (with missing edges added and spurious edges deleted).

        The values in ``entropy_args`` are passed to
        :meth:`~graph_tool.inference.blockmodel.BlockState.entropy()` to
        calculate the log-probability.

        """


        Si = self.entropy(**dict(dict(partition_dl=False),
                                 **entropy_args))

        pos = {}
        for u, v in itertools.chain(missing, spurious):
            pos[u] = self.b[u]
            pos[v] = self.b[v]

        self.remove_vertex(pos.keys())

        try:
            new_es = []
            for u, v in missing:
                if not self.is_weighted:
                    e = self.g.add_edge(u, v)
                else:
                    e = self.g.edge(u, v)
                    if e is None:
                        e = self.g.add_edge(u, v)
                        self.eweight[e] = 0
                    self.eweight[e] += 1
                new_es.append(e)

            old_es = []
            for e in spurious:
                u, v = e
                if isinstance(e, tuple):
                    e = self.g.edge(u, v)
                    if e is None:
                        raise ValueError("edge not found: (%d, %d)" % (int(u),
                                                                       int(v)))

                if self.is_weighted:
                    self.eweight[e] -= 1
                    if self.eweight[e] == 0:
                        self.g.remove_edge(e)
                else:
                    self.g.remove_edge(e)
                old_es.append((u, v))

            self.add_vertex(pos.keys(), pos.values())

            Sf = self.entropy(**dict(dict(partition_dl=False),
                                     **entropy_args))

            self.remove_vertex(pos.keys())

        finally:
            if self.is_weighted:
                for e in reversed(new_es):
                    self.eweight[e] -= 1
                    if self.eweight[e] == 0:
                        self.g.remove_edge(e)
            else:
                for e in reversed(new_es):
                    self.g.remove_edge(e)
            for u, v in old_es:
                if self.is_weighted:
                    e = self.g.edge(u, v)
                    if e is None:
                        e = self.g.add_edge(u, v)
                        self.eweight[e] = 0
                    self.eweight[e] += 1
                else:
                    self.g.add_edge(u, v)

            self.add_vertex(pos.keys(), pos.values())

        L = Si - Sf

        if _bm_test():
            state = self.copy()
            set_test(False)
            L_alt = state.get_edges_prob(edge_list, missing=missing,
                                         entropy_args=entropy_args)
            set_test(True)
            assert math.isclose(L, L_alt, abs_tol=1e-8), \
                "inconsistent missing=%s edge probability (%g, %g): %s, %s" % \
                (str(missing), L, L_alt,  str(entropy_args), str(edge_list))

        return L

    def _clear_egroups(self):
        self._state.clear_egroups()

    def _mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.mcmc_sweep(mcmc_state, self._state,
                                       _get_rng())

    def _mcmc_sweep_parallel_dispatch(states, mcmc_states):
        return libinference.mcmc_sweep_parallel(mcmc_states,
                                                [s._state for s in states],
                                                _get_rng())

    def mcmc_sweep(self, beta=1., c=1., d=.01, niter=1, entropy_args={},
                   allow_vacate=True, sequential=True, deterministic=False,
                   vertices=None, verbose=False, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC to sample network partitions.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``1.``)
            Inverse temperature.
        c : ``float`` (optional, default: ``1.``)
            Sampling parameter ``c`` for move proposals: For :math:`c\to 0` the
            blocks are sampled according to the local neighborhood of a given
            node and their block connections; for :math:`c\to\infty` the blocks
            are sampled randomly. Note that only for :math:`c > 0` the MCMC is
            guaranteed to be ergodic.
        d : ``float`` (optional, default: ``.01``)
            Probability of selecting a new (i.e. empty) group for a given move.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps to perform. During each sweep, a move attempt is
            made for each node.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`graph_tool.inference.blockmodel.BlockState.entropy`.
        allow_vacate : ``bool`` (optional, default: ``True``)
            Allow groups to be vacated.
        sequential : ``bool`` (optional, default: ``True``)
            If ``sequential == True`` each vertex move attempt is made
            sequentially, where vertices are visited in random order. Otherwise
            the moves are attempted by sampling vertices randomly, so that the
            same vertex can be moved more than once, before other vertices had
            the chance to move.
        deterministic : ``bool`` (optional, default: ``False``)
            If ``sequential == True`` and ``deterministic == True`` the
            vertices will be visited in deterministic order.
        vertices : ``list`` of ints (optional, default: ``None``)
            If provided, this should be a list of vertices which will be
            moved. Otherwise, all vertices will.
        verbose : ``bool`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of vertex moves attempted.
        nmoves : ``int``
            Number of vertices moved.

        Notes
        -----
        This algorithm has an :math:`O(E)` complexity, where :math:`E` is the
        number of edges (independent of the number of blocks).

        References
        ----------
        .. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
           greedy heuristic for the inference of stochastic block models", Phys.
           Rev. E 89, 012804 (2014), :doi:`10.1103/PhysRevE.89.012804`,
           :arxiv:`1310.4378`
        """

        mcmc_state = DictState(locals())
        entropy_args = dict(self._entropy_args, **entropy_args)
        if (_bm_test() and entropy_args["multigraph"] and
            not entropy_args["dense"] and
            hasattr(self, "degs") and
            not isinstance(self.degs, libinference.simple_degs_t)):
            entropy_args["multigraph"] = False
        mcmc_state.oentropy_args = get_entropy_args(entropy_args)
        mcmc_state.vlist = Vector_size_t()
        if vertices is None:
            vertices = self.g.vertex_index.copy().fa
            if self.is_weighted:
                # ignore vertices with zero weight
                vw = self.vweight.fa
                vertices = vertices[vw > 0]
        mcmc_state.vlist.resize(len(vertices))
        mcmc_state.vlist.a = vertices
        mcmc_state.E = self.get_E()
        mcmc_state.state = self._state

        dispatch = kwargs.pop("dispatch", True)
        test = kwargs.pop("test", True)
        if dispatch:
            if _bm_test() and test:
                assert self._check_clabel(), "invalid clabel before sweep"
                Si = self.entropy(**entropy_args)
            try:
                dS, nattempts, nmoves = self._mcmc_sweep_dispatch(mcmc_state)
            finally:
                self.B = self.bg.num_vertices()

            if _bm_test() and test:
                Sff = self.entropy(**dmask(entropy_args, ["callback"]))
                assert self._check_clabel(), "invalid clabel after sweep"
                Sf = self.entropy(**entropy_args)
                assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                    "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                                str(entropy_args))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        if not dispatch:
            return mcmc_state

        return dS, nattempts, nmoves

    def _multiflip_mcmc_sweep_dispatch(self, mcmc_state):
        return libinference.multiflip_mcmc_sweep(mcmc_state, self._state,
                                                 _get_rng())

    def _multiflip_mcmc_sweep_parallel_dispatch(states, mcmc_states):
        return libinference.multiflip_mcmc_sweep_parallel(mcmc_states,
                                                          [s._state for s in states],
                                                          _get_rng())


    def multiflip_mcmc_sweep(self, beta=1., c=1., psingle=None, psplit=1,
                             pmerge=1, pmergesplit=1, d=0.01, gibbs_sweeps=10,
                             niter=1, entropy_args={}, accept_stats=None,
                             verbose=False, **kwargs):
        r"""Perform ``niter`` sweeps of a Metropolis-Hastings acceptance-rejection
        sampling MCMC with multiple simultaneous moves to sample network
        partitions.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``1.``)
            Inverse temperature.
        c : ``float`` (optional, default: ``1.``)
            Sampling parameter ``c`` for move proposals: For :math:`c\to 0` the
            blocks are sampled according to the local neighborhood of a given
            node and their block connections; for :math:`c\to\infty` the blocks
            are sampled randomly. Note that only for :math:`c > 0` the MCMC is
            guaranteed to be ergodic.
        psingle : ``float`` (optional, default: ``None``)
            Relative probability of proposing a single node move. If ``None``,
            it will be selected as the number of nodes in the graph.
        psplit : ``float`` (optional, default: ``1``)
            Relative probability of proposing a group split.
        pmergesplit : ``float`` (optional, default: ``1``)
            Relative probability of proposing a marge-split move.
        d : ``float`` (optional, default: ``1``)
            Probability of selecting a new (i.e. empty) group for a given
            single-node move.
        gibbs_sweeps : ``int`` (optional, default: ``10``)
            Number of sweeps of Gibbs sampling to be performed (i.e. each node
            is attempted once per sweep) to refine a split proposal.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps to perform. During each sweep, a move attempt is
            made for each node, on average.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`graph_tool.inference.blockmodel.BlockState.entropy`.
        verbose : ``bool`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of vertex moves attempted.
        nmoves : ``int``
            Number of vertices moved.

        Notes
        -----
        This algorithm has an :math:`O(E)` complexity, where :math:`E` is the
        number of edges (independent of the number of blocks).

        """

        if psingle is None:
            psingle = self.g.num_vertices()
        gibbs_sweeps = max(gibbs_sweeps, 1)
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        entropy_args = dict(self._entropy_args, **entropy_args)
        if (_bm_test() and entropy_args["multigraph"] and
            not entropy_args["dense"] and
            hasattr(self, "degs") and
            not isinstance(self.degs, libinference.simple_degs_t)):
            entropy_args["multigraph"] = False
        mcmc_state.oentropy_args = get_entropy_args(entropy_args)
        mcmc_state.state = self._state

        dispatch = kwargs.pop("dispatch", True)
        test = kwargs.pop("test", True)
        if dispatch:
            if _bm_test() and test:
                assert self._check_clabel(), "invalid clabel before sweep"
                Si = self.entropy(**entropy_args)

            try:
                dS, nattempts, nmoves = self._multiflip_mcmc_sweep_dispatch(mcmc_state)
            finally:
                self.B = self.bg.num_vertices()

            if _bm_test() and test:
                assert self._check_clabel(), "invalid clabel after sweep"
                Sf = self.entropy(**entropy_args)
                assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                    "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                                str(entropy_args))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        if not dispatch:
            return mcmc_state

        if accept_stats is not None:
            for key in ["proposal", "acceptance"]:
                if key not in accept_stats:
                    accept_stats[key] = numpy.zeros(len(nproposal),
                                                    dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return dS, nattempts, nmoves

    def _gibbs_sweep_dispatch(self, gibbs_state):
        return libinference.gibbs_sweep(gibbs_state, self._state,
                                        _get_rng())

    def _gibbs_sweep_parallel_dispatch(states, gibbs_states):
        return libinference.gibbs_sweep_parallel(gibbs_states,
                                                 [s._state for s in states],
                                                 _get_rng())

    def gibbs_sweep(self, beta=1., niter=1, entropy_args={},
                    allow_new_group=True, sequential=True, deterministic=False,
                    vertices=None, verbose=False, **kwargs):
        r"""Perform ``niter`` sweeps of a rejection-free Gibbs sampling MCMC
        to sample network partitions.

        Parameters
        ----------
        beta : ``float`` (optional, default: ``1.``)
            Inverse temperature.
        niter : ``int`` (optional, default: ``1``)
            Number of sweeps to perform. During each sweep, a move attempt is
            made for each node.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`graph_tool.inference.blockmodel.BlockState.entropy`.
        allow_new_group : ``bool`` (optional, default: ``True``)
            Allow the number of groups to increase and decrease.
        sequential : ``bool`` (optional, default: ``True``)
            If ``sequential == True`` each vertex move attempt is made
            sequentially, where vertices are visited in random order. Otherwise
            the moves are attempted by sampling vertices randomly, so that the
            same vertex can be moved more than once, before other vertices had
            the chance to move.
        deterministic : ``bool`` (optional, default: ``False``)
            If ``sequential == True`` and ``deterministic == True`` the
            vertices will be visited in deterministic order.
        vertices : ``list`` of ints (optional, default: ``None``)
            If provided, this should be a list of vertices which will be
            moved. Otherwise, all vertices will.
        verbose : ``bool`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of vertex moves attempted.
        nmoves : ``int``
            Number of vertices moved.

        Notes
        -----
        This algorithm has an :math:`O(E\times B)` complexity, where :math:`B`
        is the number of blocks, and :math:`E` is the number of edges.

        """

        gibbs_state = DictState(locals())
        entropy_args = dict(self._entropy_args, **entropy_args)
        gibbs_state.oentropy_args = get_entropy_args(entropy_args)
        gibbs_state.vlist = Vector_size_t()
        if vertices is None:
            vertices = self.g.get_vertices()
        gibbs_state.vlist.resize(len(vertices))
        gibbs_state.vlist.a = vertices
        gibbs_state.E = self.get_E()
        gibbs_state.state = self._state

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            assert self._check_clabel(), "invalid clabel before sweep"
            Si = self.entropy(**entropy_args)

        try:
            dS, nattempts, nmoves = self._gibbs_sweep_dispatch(gibbs_state)
        finally:
            self.B = self.bg.num_vertices()

        if _bm_test() and test:
            assert self._check_clabel(), "invalid clabel after sweep"
            Sf = self.entropy(**entropy_args)
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                            str(entropy_args))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        return dS, nattempts, nmoves

    def _multicanonical_sweep_dispatch(self, multicanonical_state):
        if multicanonical_state.multiflip:
            return libinference.multicanonical_multiflip_sweep(multicanonical_state,
                                                               self._state,
                                                               _get_rng())
        else:
            return libinference.multicanonical_sweep(multicanonical_state,
                                                     self._state, _get_rng())

    def multicanonical_sweep(self, m_state, multiflip=False, **kwargs):
        r"""Perform ``niter`` sweeps of a non-Markovian multicanonical sampling using the
        Wang-Landau algorithm.

        Parameters
        ----------
        m_state : :class:`~graph_tool.inference.mcmc.MulticanonicalState`
            :class:`~graph_tool.inference.mcmc.MulticanonicalState` instance
            containing the current state of the Wang-Landau run.
        multiflip : ``bool`` (optional, default: ``False``)
            If ``True``, ``multiflip_mcmc_sweep()`` will be used, otherwise
            ``mcmc_sweep()``.
        **kwargs : Keyword parameter list
            The remaining parameters will be passed to
            ``multiflip_mcmc_sweep()`` or ``mcmc_sweep()``.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of vertex moves attempted.
        nmoves : ``int``
            Number of vertices moved.

        Notes
        -----
        This algorithm has an :math:`O(E)` complexity, where :math:`E` is the
        number of edges (independent of the number of blocks).

        References
        ----------
        .. [wang-efficient-2001] Fugao Wang, D. P. Landau, "An efficient, multiple
           range random walk algorithm to calculate the density of states", Phys.
           Rev. Lett. 86, 2050 (2001), :doi:`10.1103/PhysRevLett.86.2050`,
           :arxiv:`cond-mat/0011174`
        """

        if not multiflip:
            kwargs["sequential"] = False
        kwargs["beta"] = 1

        args = dmask(locals(), ["self", "kwargs"])
        multi_state = DictState(args)

        entropy_args = kwargs.get("entropy_args", {})

        if multiflip:
            mcmc_state = self.multiflip_mcmc_sweep(dispatch=False, **kwargs)
        else:
            mcmc_state = self.mcmc_sweep(dispatch=False, **kwargs)

        multi_state.update(mcmc_state)
        multi_state.multiflip = multiflip

        multi_state.S = self.entropy(**entropy_args)
        multi_state.state = self._state

        multi_state.f = m_state._f
        multi_state.S_min = m_state._S_min
        multi_state.S_max = m_state._S_max
        multi_state.hist = m_state._hist
        multi_state.dens = m_state._density

        if (multi_state.S < multi_state.S_min or
            multi_state.S > multi_state.S_max):
            raise ValueError("initial entropy %g out of bounds (%g, %g)" %
                             (multi_state.S, multi_state.S_min,
                              multi_state.S_max))

        try:
            S, nattempts, nmoves = self._multicanonical_sweep_dispatch(multi_state)
        finally:
            self.B = self.bg.num_vertices()

        if _bm_test() and kwargs.pop("test", True):
            assert self._check_clabel(), "invalid clabel after sweep"
            Sf = self.entropy(**entropy_args)
            assert math.isclose(S, Sf, abs_tol=1e-8), \
                "inconsistent entropy after sweep %g (%g): %s" % \
                (S, Sf, str(entropy_args))

        return S, nattempts, nmoves

    def _multicanonical_B_sweep_dispatch(self, multicanonical_state):
        return libinference.multicanonical_B_multiflip_sweep(multicanonical_state,
                                                             self._state,
                                                             _get_rng())

    def multicanonical_B_sweep(self, m_state, **kwargs):
        r"""Perform ``niter`` sweeps of a non-Markovian multicanonical sampling using the
        Wang-Landau algorithm.

        Parameters
        ----------
        m_state : :class:`~graph_tool.inference.mcmc.MulticanonicalState`
            :class:`~graph_tool.inference.mcmc.MulticanonicalState` instance
            containing the current state of the Wang-Landau run.
        multiflip : ``bool`` (optional, default: ``False``)
            If ``True``, ``multiflip_mcmc_sweep()`` will be used, otherwise
            ``mcmc_sweep()``.
        **kwargs : Keyword parameter list
            The remaining parameters will be passed to
            ``multiflip_mcmc_sweep()`` or ``mcmc_sweep()``.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of vertex moves attempted.
        nmoves : ``int``
            Number of vertices moved.

        Notes
        -----
        This algorithm has an :math:`O(E)` complexity, where :math:`E` is the
        number of edges (independent of the number of blocks).

        References
        ----------
        .. [wang-efficient-2001] Fugao Wang, D. P. Landau, "An efficient, multiple
           range random walk algorithm to calculate the density of states", Phys.
           Rev. Lett. 86, 2050 (2001), :doi:`10.1103/PhysRevLett.86.2050`,
           :arxiv:`cond-mat/0011174`
        """

        kwargs["beta"] = 1

        args = dmask(locals(), ["self", "kwargs"])
        multi_state = DictState(args)

        entropy_args = kwargs.get("entropy_args", {})

        mcmc_state = self.multiflip_mcmc_sweep(dispatch=False, **kwargs)

        multi_state.update(mcmc_state)

        multi_state.S = self.entropy(**entropy_args)
        multi_state.state = self._state

        multi_state.f = m_state._f
        multi_state.B_min = m_state._B_min
        multi_state.B_max = m_state._B_max
        multi_state.hist = m_state._hist
        multi_state.dens = m_state._density

        B = self.get_nonempty_B()
        if (B < multi_state.B_min or B > multi_state.B_max):
            raise ValueError("initial number of groups %d out of bounds (%d, %d)" %
                             (B, multi_state.B_min, multi_state.B_max))

        try:
            S, nattempts, nmoves = self._multicanonical_B_sweep_dispatch(multi_state)
        finally:
            self.B = self.bg.num_vertices()

        if _bm_test() and kwargs.pop("test", True):
            assert self._check_clabel(), "invalid clabel after sweep"
            Sf = self.entropy(**entropy_args)
            assert math.isclose(S, Sf, abs_tol=1e-8), \
                "inconsistent entropy after sweep %g (%g): %s" % \
                (S, Sf, str(entropy_args))

        return S, nattempts, nmoves

    def _exhaustive_sweep_dispatch(self, exhaustive_state, callback, hist):
        if callback is not None:
            return libinference.exhaustive_sweep(exhaustive_state, self._state,
                                                 callback)
        else:
            if hist is None:
                return libinference.exhaustive_sweep_iter(exhaustive_state,
                                                          self._state)
            else:
                return libinference.exhaustive_dens(exhaustive_state,
                                                    self._state, hist[0],
                                                    hist[1], hist[2])

    def exhaustive_sweep(self, entropy_args={}, callback=None, density=None,
                         vertices=None, initial_partition=None, max_iter=None):
        r"""Perform an exhaustive loop over all possible network partitions.

        Parameters
        ----------
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`graph_tool.inference.blockmodel.BlockState.entropy`.
        callback : callable object (optional, default: ``None``)
            Function to be called for each partition, with three arguments ``(S,
            S_min, b_min)`` corresponding to the the current entropy value, the
            minimum entropy value so far, and the corresponding partition,
            respectively. If not provided, and ``hist is None`` an iterator over
            the same values will be returned instead.
        density : ``tuple`` (optional, default: ``None``)
            If provided, it should contain a tuple with values ``(S_min, S_max,
            n_bins)``, which will be used to obtain the density of states via a
            histogram of size ``n_bins``. This parameter is ignored unless
            ``callback is None``.
        vertices : iterable of ints (optional, default: ``None``)
            If provided, this should be a list of vertices which will be
            moved. Otherwise, all vertices will.
        initial_partition : iterable  of ints (optional, default: ``None``)
            If provided, this will provide the initial partition for the
            iteration.
        max_iter : ``int`` (optional, default: ``None``)
            If provided, this will limit the total number of iterations.

        Returns
        -------
        states : iterator over (S, S_min, b_min)
            If ``callback`` is ``None`` and ``hist`` is ``None``, the function
            will return an iterator over ``(S, S_min, b_min)`` corresponding to
            the the current entropy value, the minimum entropy value so far, and
            the corresponding partition, respectively.
        Ss, counts : pair of :class:`numpy.ndarray`
            If ``callback is None`` and ``hist is not None``, the function will
            return the values of each bin (``Ss``) and the state count of each
            bin (``counts``).
        b_min : :class:`~graph_tool.VertexPropertyMap`
            If ``callback is not None`` or ``hist is not None``, the function
            will also return partition with smallest entropy.

        Notes
        -----

        This algorithm has an :math:`O(B^N)` complexity, where :math:`B` is the
        number of blocks, and :math:`N` is the number of vertices.

        """

        exhaustive_state = DictState(dict(max_iter=max_iter if max_iter is not None else 0))
        entropy_args = dict(self._entropy_args, **entropy_args)
        exhaustive_state.oentropy_args = get_entropy_args(entropy_args)
        exhaustive_state.vlist = Vector_size_t()
        if vertices is None:
            vertices = self.g.vertex_index.copy().fa
            if self.is_weighted:
                # ignore vertices with zero weight
                vw = self.vweight.fa
                vertices = vertices[vw > 0]
        if initial_partition is None:
            initial_partition = zeros(len(vertices), dtype="uint64")
        self.move_vertex(vertices, initial_partition)
        exhaustive_state.vlist.resize(len(vertices))
        exhaustive_state.vlist.a = vertices
        exhaustive_state.S = self.entropy(**entropy_args)
        exhaustive_state.state = self._state
        exhaustive_state.b_min = b_min = self.g.new_vp("int32_t")

        if density is not None:
            density = (density[0], density[1],
                       numpy.zeros(density[2], dtype="uint64"))
        if callback is not None:
            _callback = lambda S, S_min: callback(S, S_min, b_min)
        else:
            _callback = None
        try:
            ret = self._exhaustive_sweep_dispatch(exhaustive_state, _callback,
                                                  density)
        finally:
            self.B = self.bg.num_vertices()
        if _callback is None:
            if density is None:
                return ((S, S_min, b_min) for S, S_min in ret)
            else:
                Ss = numpy.linspace(density[0], density[1], len(density[2]))
                return (Ss, density[2]), b_min
        else:
            return b_min

    def _merge_sweep_dispatch(self, merge_state):
        if not self.is_weighted:
            raise ValueError("state must be weighted to perform merges")

        if (merge_state.oentropy_args.multigraph and
            not merge_state.oentropy_args.dense):
            raise ValueError("can only merge multigraphs if dense == True")

        return libinference.merge_sweep(merge_state, self._state,
                                        _get_rng())

    def merge_sweep(self, nmerges=1, niter=10, entropy_args={}, parallel=True,
                    verbose=False, **kwargs):
        r"""Perform ``niter`` merge sweeps, where block nodes are progressively
        merged together in a manner that least increases the entropy.

        Parameters
        ----------
        nmerges : ``int`` (optional, default: ``1``)
            Number block nodes to merge.
        niter : ``int`` (optional, default: ``1``)
            Number of merge attempts to perform for each block node, before the
            best one is selected.
        entropy_args : ``dict`` (optional, default: ``{}``)
            Entropy arguments, with the same meaning and defaults as in
            :meth:`graph_tool.inference.blockmodel.BlockState.entropy`.
        parallel : ``bool`` (optional, default: ``True``)
            If ``parallel == True``, the merge candidates are obtained in
            parallel.
        verbose : ``bool`` (optional, default: ``False``)
            If ``verbose == True``, detailed information will be displayed.

        Notes
        -----

        This function should only be called for block states, obtained from
        :meth:`graph_tool.inference.blockmodel.BlockState.get_block_state`.

        Returns
        -------
        dS : ``float``
            Entropy difference after the sweeps.
        nattempts : ``int``
            Number of attempted merges.
        nmoves : ``int``
            Number of vertices merged.

        References
        ----------
        .. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
           greedy heuristic for the inference of stochastic block models", Phys.
           Rev. E 89, 012804 (2014), :doi:`10.1103/PhysRevE.89.012804`,
           :arxiv:`1310.4378`
        """

        merge_state = DictState(locals())
        merge_state.E = self.get_E()
        merge_state.state = self._state
        entropy_args = dict(self._entropy_args, **entropy_args)
        merge_state.entropy_args = get_entropy_args(entropy_args)
        merge_state.oentropy_args = merge_state.entropy_args

        if _bm_test():
            assert self._check_clabel(), "invalid clabel before sweep"
            Si = self.entropy(**entropy_args)

        dS, nattempts, nmoves = self._merge_sweep_dispatch(merge_state)

        if _bm_test():
            assert self._check_clabel(), "invalid clabel after sweep"
            Sf = self.entropy(**entropy_args)
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                            str(entropy_args))

        return dS, nattempts, nmoves

    def shrink(self, B, **kwargs):
        """Reduces the order of current state by progressively merging groups, until
        only ``B`` are left. All remaining keyword arguments are passed to
        :meth:`graph_tool.inference.blockmodel.BlockState.merge_sweep`.

        This function leaves the current state untouched and returns instead a
        copy with the new partition.
        """
        bstate = self.get_block_state(vweight=True,
                                      clabel=self.get_bclabel(),
                                      deg_corr=self.deg_corr)

        nB = bstate.get_nonempty_B()
        while nB > B:
            bstate.merge_sweep(nB - B, **kwargs)
            nB = bstate.get_nonempty_B()
        b = self.b.copy()
        pmap(b, bstate.merge_map)
        continuous_map(b)
        state = self.copy(b=b, Lrecdx=bstate.Lrecdx)
        if _bm_test():
            nB = (state.wr.a > 0).sum()
            assert nB == B, "wrong number of blocks after shrink: %d (should be %d)" % (nB, B)
        return state

    def collect_edge_marginals(self, p=None, update=1):
        r"""Collect the edge marginal histogram, which counts the number of times
        the endpoints of each node have been assigned to a given block pair.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.blockmodel.BlockState.mcmc_sweep` function.

        Parameters
        ----------
        p : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            Edge property map with edge marginals to be updated.  If not
            provided, an empty histogram will be created.
        update : float (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        p : :class:`~graph_tool.EdgePropertyMap`
            Edge property map with updated edge marginals.

        Examples
        --------
        .. testsetup:: collect_edge_marginals

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: collect_edge_marginals

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=4, deg_corr=True)
           >>> pe = None
           >>> state.mcmc_sweep(niter=1000)   # remove part of the transient
           (...)
           >>> for i in range(1000):
           ...     ret = state.mcmc_sweep(niter=10)
           ...     pe = state.collect_edge_marginals(pe)
           >>> gt.bethe_entropy(g, pe)[0]
           1.7733496...
        """

        if p is None:
            p = self.g.new_ep("python::object",
                              vals=[libinference.BlockPairHist()
                                    for i in range(self.g.num_edges())])

        libinference.edge_marginals(self.g._Graph__graph,
                                    _prop("v", self.g, self.b),
                                    _prop("e", self.g, p),
                                    update)
        return p

    def collect_vertex_marginals(self, p=None, b=None, unlabel=False, update=1):
        r"""Collect the vertex marginal histogram, which counts the number of times a
        node was assigned to a given block.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.blockmodel.BlockState.mcmc_sweep` function.

        Parameters
        ----------
        p : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with vector-type values, storing the previous block
            membership counts. If not provided, an empty histogram will be created.
        b : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            Vertex property map with group partition. If not provided, the
            state's partition will be used.
        unlabel : bool (optional, default: ``False``)
            If ``True``, a canonical labelling of the groups will be used, so
            that each partition is uniquely represented.
        update : int (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.

        Returns
        -------
        p : :class:`~graph_tool.VertexPropertyMap`
            Vertex property map with vector-type values, storing the accumulated
            block membership counts.

        Examples
        --------
        .. testsetup:: cvm

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: cvm

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=4, deg_corr=True)
           >>> pv = None
           >>> state.mcmc_sweep(niter=1000)   # remove part of the transient
           (...)
           >>> for i in range(1000):
           ...     ret = state.mcmc_sweep(niter=10)
           ...     pv = state.collect_vertex_marginals(pv)
           >>> gt.mf_entropy(g, pv)
           22.237735...
           >>> gt.graph_draw(g, pos=g.vp["pos"], vertex_shape="pie",
           ...               vertex_pie_fractions=pv, output="polbooks_blocks_soft_B4.svg")
           <...>

        .. figure:: polbooks_blocks_soft_B4.*
           :align: center

           "Soft" block partition of a political books network with :math:`B=4`.

        """

        if p is None:
            p = self.g.new_vp("vector<int>")
        if b is None:
            b = self.b
        if unlabel:
            b = perfect_prop_hash([b])[0]
        libinference.vertex_marginals(self.g._Graph__graph,
                                      _prop("v", self.g, b),
                                      _prop("v", self.g, p),
                                      update)
        return p

    def collect_partition_histogram(self, h=None, update=1, unlabel=True):
        r"""Collect a histogram of partitions.

        This should be called multiple times, e.g. after repeated runs of the
        :meth:`graph_tool.inference.blockmodel.BlockState.mcmc_sweep` function.

        Parameters
        ----------
        h : :class:`~graph_tool.inference.blockmodel.PartitionHist` (optional, default: ``None``)
            Partition histogram. If not provided, an empty histogram will be created.
        update : float (optional, default: ``1``)
            Each call increases the current count by the amount given by this
            parameter.
        unlabel : bool (optional, default: ``True``)
            If ``True``, a canonical labelling of the groups will be used, so
            that each partition is uniquely represented.

        Returns
        -------
        h : :class:`~graph_tool.inference.blockmodel.PartitionHist` (optional, default: ``None``)
            Updated Partition histogram.

        Examples
        --------
        .. testsetup:: cvm

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: cvm

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.BlockState(g, B=4, deg_corr=True)
           >>> ph = None
           >>> state.mcmc_sweep(niter=1000)   # remove part of the transient
           (...)
           >>> for i in range(1000):
           ...     ret = state.mcmc_sweep(niter=10)
           ...     ph = state.collect_partition_histogram(ph)
           >>> gt.microstate_entropy(ph)
           132.254124...
        """

        if h is None:
            h = PartitionHist()
        libinference.collect_partitions(_prop("v", self.g, self.b),
                                        h, update, unlabel)
        return h

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.graph_draw` that
        draws the state of the graph as colors on the vertices and edges."""
        gradient = self.g.new_ep("double")
        gradient = group_vector_property([gradient])
        from graph_tool.draw import graph_draw
        return graph_draw(self.g,
                          vertex_fill_color=kwargs.get("vertex_fill_color",
                                                       self.b),
                          vertex_color=kwargs.get("vertex_color", self.b),
                          edge_gradient=kwargs.get("edge_gradient",
                                                   gradient),
                          **dmask(kwargs, ["vertex_fill_color",
                                           "vertex_color",
                                           "edge_gradient"]))

    def sample_graph(self, canonical=False, multigraph=True, self_loops=True,
                     sample_params=False, max_ent=False, n_iter=1000):
        r"""Sample a new graph from the fitted model.

        Parameters
        ----------
        canonical : ``bool`` (optional, default: ``False``)
            If ``canonical == True``, the graph will be sampled from the
            maximum-likelihood estimate of the canonical stochastic block
            model. Otherwise, it will be sampled from the microcanonical model.
        multigraph : ``bool`` (optional, default: ``True``)
            If ``True``, parallel edges will be allowed.
        self-loops : ``bool`` (optional, default: ``True``)
            If ``True``, self-loops will be allowed.
        sample_params : ``bool`` (optional, default: ``True``)
            If ``True``, and ``canonical == True`` and ``max_ent == False``,
            the count parameters (edges between groups and node degrees) will be
            sampled from their posterior distribution conditioned on the actual
            state. Otherwise, their maximum-likelihood values will be used.
        max_ent : ``bool`` (optional, default: ``False``)
            If ``True``, maximum-entropy model variants will be used.
        n_iter : ``int`` (optional, default: ``1000``)
            Number of iterations used (only relevant if ``canonical == False``
            and ``max_ent == True``).

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            Generated graph.

        Notes
        -----
        This function is just a convenience wrapper to
        :func:`~graph_tool.generation.generate_sbm`. However, if
        ``max_ent==True`` and ``canonical == False`` it wraps
        :func:`~graph_tool.generation.random_rewire` instead.

        Examples
        --------
        .. doctest:: gen_sbm

           >>> g = gt.collection.data["polbooks"]
           >>> state = gt.minimize_blockmodel_dl(g, B_max=3)
           >>> u = state.sample_graph(canonical=True, self_loops=False, multigraph=False)
           >>> ustate = gt.BlockState(u, b=state.b)
           >>> state.draw(pos=g.vp.pos, output="polbooks-sbm.svg")
           <...>
           >>> ustate.draw(pos=u.own_property(g.vp.pos), output="polbooks-sbm-sampled.svg")
           <...>

        .. image:: polbooks-sbm.svg
            :width: 40%
        .. image:: polbooks-sbm-sampled.svg
            :width: 40%

        *Left:* Political books network. *Right:* Sample from the degree-corrected
        SBM fitted to the original network.

        """
        in_degs = out_degs = None
        if isinstance(self.eweight, libinference.unity_eprop_t):
            eweight = None
        else:
            eweight = self.eweight
        if self.deg_corr:
            out_degs = self.g.degree_property_map("out", weight=eweight).fa
            if self.g.is_directed():
                in_degs = self.g.degree_property_map("in", weight=eweight).fa
            else:
                in_degs = None
        probs = adjacency(self.bg, weight=self.mrs).T
        if not max_ent:
            if canonical and sample_params:
                rs = self.wr.a > 0
                B = rs.sum()
                if self.g.is_directed():
                    p = self.g.num_edges() / B ** 2
                if not self.g.is_directed():
                    p = 2 * self.g.num_edges() / ((B + 1) * B)
                idx = probs.nonzero()
                probs[idx] = numpy.random.gamma(probs[idx] + 1, p/(p + 1))
                for r in rs:
                    idx = self.b.fa == r
                    er = probs[r,:].sum()
                    out_degs[idx] = numpy.random.dirichlet(out_degs[idx] + 1)
                    out_degs[idx] = numpy.random.multinomial(int(er), out_degs[idx])
                    if in_degs is not None:
                        er = probs[:,r].sum()
                        in_degs[idx] = numpy.random.dirichlet(in_degs[idx] + 1)
                        in_degs[idx] = numpy.random.multinomial(int(er), in_degs[idx])

            g = generate_sbm(b=self.b.fa, probs=probs,
                             in_degs=in_degs, out_degs=out_degs,
                             directed=self.g.is_directed(),
                             micro_ers=not canonical,
                             micro_degs=not canonical and self.deg_corr)

            if not multigraph:
                remove_parallel_edges(g)
            if not self_loops:
                remove_self_loops(g)
        else:
            if canonical:
                ret = solve_sbm_fugacities(self.b.fa, probs, out_degs, in_degs,
                                           multigraph=multigraph,
                                           self_loops=self_loops)
                if in_degs is None:
                    mrs, theta_out = ret
                    theta_in = None
                else:
                    mrs, theta_out, theta_in = ret
                g = generate_maxent_sbm(self.b.fa, mrs, theta_out, theta_in,
                                        directed=self.g.is_directed(),
                                        multigraph=multigraph,
                                        self_loops=self_loops)
            else:
                g = self.g.copy()
                if self.deg_corr:
                    random_rewire(g, model="constrained-configuration",
                                  block_membership=g.own_property(self.b),
                                  configuration=False, parallel_edges=multigraph,
                                  self_loops=self_loops, n_iter=n_iter)
                else:
                    random_rewire(g, model="blockmodel-micro",
                                  block_membership=g.own_property(self.b),
                                  configuration=False, parallel_edges=multigraph,
                                  self_loops=self_loops, n_iter=n_iter)
        return g

def model_entropy(B, N, E, directed=False, nr=None):
    r"""Computes the amount of information necessary for the parameters of the
    traditional blockmodel ensemble, for ``B`` blocks, ``N`` vertices, ``E``
    edges, and either a directed or undirected graph.

    This is equivalently defined as minus the log-likelihood of sampling the
    parameters from a nonparametric generative model.

    A traditional blockmodel is defined as a set of :math:`N` vertices which can
    belong to one of :math:`B` blocks, and the matrix :math:`e_{rs}` describes
    the number of edges from block :math:`r` to :math:`s` (or twice that number
    if :math:`r=s` and the graph is undirected).

    For an undirected graph, the number of distinct :math:`e_{rs}` matrices is
    given by,

    .. math::

       \Omega_m = \left(\!\!{B(B+1)/2 \choose E}\!\!\right)

    and for a directed graph,

    .. math::
       \Omega_m = \left(\!\!{B^2 \choose E}\!\!\right)


    where :math:`\left(\!{n \choose k}\!\right) = {n+k-1\choose k}` is the
    number of :math:`k` combinations with repetitions from a set of size
    :math:`n`. Hence, we have the description length of the edge counts

    .. math::

        -\ln P(\boldsymbol{e}) = \ln \Omega_m.

    For the node partition :math:`\boldsymbol{b}` we assume a two-level Bayesian
    hierarchy, where first the group size histogram is generated, and
    conditioned on it the partition, which leads to a description length:

    .. math::

       -\ln P(\boldsymbol{b}) = \ln {N - 1 \choose B - 1} + \ln N! - \sum_r \ln n_r!.

    where :math:`n_r` is the number of nodes in block :math:`r`.

    The total information necessary to describe the model is then,

    .. math::

       -\ln P(\boldsymbol{e}, \boldsymbol{b}) = -\ln P(\boldsymbol{e}) - \ln P(\boldsymbol{b}).

    If ``nr`` is ``None``, it is assumed :math:`n_r=N/B`. If ``nr`` is
    ``False``, the partition term :math:`-\ln P(\boldsymbol{b})` is omitted
    entirely.

    References
    ----------

    .. [peixoto-parsimonious-2013] Tiago P. Peixoto, "Parsimonious module
       inference in large networks", Phys. Rev. Lett. 110, 148701 (2013),
       :doi:`10.1103/PhysRevLett.110.148701`, :arxiv:`1212.4794`.
    .. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric
       Bayesian inference of the microcanonical stochastic block model",
       Phys. Rev. E 95 012317 (2017), :doi:`10.1103/PhysRevE.95.012317`,
       :arxiv:`1610.02703`
    """

    if directed:
        x = (B * B);
    else:
        x = (B * (B + 1)) / 2;
    if nr is False:
        L = lbinom(x + E - 1, E)
    else:
        L = lbinom(x + E - 1, E) + partition_entropy(B, N, nr)
    return L

def bethe_entropy(g, p):
    r"""Compute the Bethe entropy given the edge block membership marginals.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    p : :class:`~graph_tool.EdgePropertyMap`
        Edge property map with edge marginals.

    Returns
    -------
    H : ``float``
        The Bethe entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__)
    Hmf : ``float``
        The "mean field" entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__),
        as would be returned by the :func:`mf_entropy` function.
    pv : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Vertex property map with vector-type values, storing the accumulated
        block membership counts. These are the node marginals, as would be
        returned by the
        :meth:`~graph_tool.inference.blockmodel.BlockState.collect_vertex_marginals`
        method.

    Notes
    -----

    The Bethe entropy is defined as,

    .. math::

        H = -\sum_{ij}A_{ij}\sum_{rs}\pi_{ij}(r,s)\ln\pi_{ij}(r,s) - \sum_i(1-k_i)\sum_r\pi_i(r)\ln\pi_i(r),

    where :math:`\pi_{ij}(r,s)` is the marginal probability that vertices
    :math:`i` and :math:`j` belong to blocks :math:`r` and :math:`s`,
    respectively, and :math:`\pi_i(r)` is the marginal probability that vertex
    :math:`i` belongs to block :math:`r`, and :math:`k_i` is the degree of
    vertex :math:`i` (or total degree for directed graphs).

    References
    ----------
    .. [mezard-information-2009] Marc Mézard, Andrea Montanari, "Information,
       Physics, and Computation", Oxford Univ Press, 2009.
       :DOI:`10.1093/acprof:oso/9780198570837.001.0001`

    """
    H = 0
    pv =  g.new_vertex_property("vector<double>")

    H, Hmf  = libinference.bethe_entropy(g._Graph__graph,
                                         _prop("e", g, p),
                                         _prop("v", g, pv))
    return H, Hmf, pv


def mf_entropy(g, p):
    r"""Compute the "mean field" entropy given the vertex block membership marginals.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        The graph.
    p : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with vector-type values, storing the accumulated block
        membership counts.

    Returns
    -------
    Hmf : ``float``
        The "mean field" entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__).

    Notes
    -----

    The "mean field" entropy is defined as,

    .. math::

        H = - \sum_{i,r}\pi_i(r)\ln\pi_i(r),

    where :math:`\pi_i(r)` is the marginal probability that vertex :math:`i`
    belongs to block :math:`r`.

    References
    ----------
    .. [mezard-information-2009] Marc Mézard, Andrea Montanari, "Information,
       Physics, and Computation", Oxford Univ Press, 2009.
       :DOI:`10.1093/acprof:oso/9780198570837.001.0001`
    """

    return libinference.mf_entropy(g._Graph__graph,
                                   _prop("v", g, p))

def microstate_entropy(h, unlabel=True):
    r"""Compute microstate entropy given a histogram of partitions.

    Parameters
    ----------
    h : :class:`~graph_tool.inference.blockmodel.PartitionHist` (optional, default: ``None``)
        Partition histogram.
    unlabel : bool (optional, default: ``True``)
        If ``True``, a canonical labelling of the groups will be used, so that
        each partition is uniquely represented. However, the entropy computed
        will still correspond to the full distribution over labelled partitions,
        where all permutations are assumed to be equally likely.

    Returns
    -------
    H : ``float``
        The microstate entropy value (in `nats <http://en.wikipedia.org/wiki/Nat_%28information%29>`__).

    Notes
    -----

    The microstate entropy is defined as,

    .. math::

        H = - \sum_{\boldsymbol b}p({\boldsymbol b})\ln p({\boldsymbol b}),

    where :math:`p({\boldsymbol b})` is observed frequency of labelled partition
    :math:`{\boldsymbol b}`.

    References
    ----------
    .. [mezard-information-2009] Marc Mézard, Andrea Montanari, "Information,
       Physics, and Computation", Oxford Univ Press, 2009.
       :DOI:`10.1093/acprof:oso/9780198570837.001.0001`

    """

    return libinference.partitions_entropy(h, unlabel)

from . overlap_blockmodel import *
