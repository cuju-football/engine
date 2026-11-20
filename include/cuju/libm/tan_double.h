// IBM Accurate Mathematical Library
// written by International Business Machines Corp.
// Copyright (C) 2001-2024 Free Software Foundation, Inc.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU  Lesser General Public License
// along with this program; if not, see <https://www.gnu.org/licenses/>.
//
// Ported to C++ for host and device from glibc 2.39 sysdeps/ieee754/dbl-64/s_tan.c, utan.h,
// utan.tbl and dla.h by the cuju Authors, 2026.
// SPDX-License-Identifier: LGPL-2.1-or-later

// Double tan of glibc 2.39's build for FMA and AVX2, s_tan-fma.c, bit for bit on host and device.
// Exact: each `fma` is a contraction of that build and nothing else fuses, `__branred` has none.
#pragma once

#include "../host_device.h"
#include "common.h"
#include "sin_double.h"

namespace cuju {
namespace libm {

// utan.tbl's `xfg` without its unused fourth column. Each row holds a point near k over 256, for k
// from 16 to 201, its tangent and its cotangent.
CUJU_TABLE_DEFINITION(
    double, TAN_TABLE,
    {
        0x1.000001e519d60p-4, 0x1.0055796c4e240p-4, 0x1.ff554628127b7p+3, 0x1.100001b1a7010p-4,
        0x1.10668aab892b7p-4, 0x1.e12c7be3fdf74p+3, 0x1.200002505e350p-4, 0x1.2079bff547824p-4,
        0x1.c65c5de853633p+3, 0x1.2fffffcdc4252p-4, 0x1.308f35eb16c68p-4, 0x1.ae5dae56be74fp+3,
        0x1.3ffffe3ff849fp-4, 0x1.40a71154999ccp-4, 0x1.98c43046b7352p+3, 0x1.4ffffedc9590fp-4,
        0x1.50c17429bdd80p-4, 0x1.8538491b5d674p+3, 0x1.6000000ee83f7p-4, 0x1.60de7da80cc21p-4,
        0x1.73724ef21a2a7p+3, 0x1.6ffffeb05ea41p-4, 0x1.70fe4b8c51beap-4, 0x1.63370fae562ffp+3,
        0x1.7ffffdc0515f7p-4, 0x1.812101db54498p-4, 0x1.545530e7eab5cp+3, 0x1.8ffffe384d7abp-4,
        0x1.9146c2a8d3727p-4, 0x1.46a33fd57f3fdp+3, 0x1.9ffffe4832347p-4, 0x1.a16fad50e1050p-4,
        0x1.39fe2c5537a96p+3, 0x1.b0000274540e3p-4, 0x1.b19be7ae68517p-4, 0x1.2e4813637e946p+3,
        0x1.bfffffebf2e9bp-4, 0x1.c1cb88369cd19p-4, 0x1.2367617aef223p+3, 0x1.d000023529045p-4,
        0x1.d1febc11d7ef7p-4, 0x1.1945fb8e43d4ep+3, 0x1.dffffd872a829p-4, 0x1.e23598ee4d6b7p-4,
        0x1.0fd0c76195d5fp+3, 0x1.effffff323b84p-4, 0x1.f2704ec9073e5p-4, 0x1.06f713020200fp+3,
        0x1.000000ce79195p-3, 0x1.01577bc30cc61p-3, 0x1.fd549d6564a88p+2, 0x1.07fffee40e918p-3,
        0x1.0978d8279ac01p-3, 0x1.edbb59294bc03p+2, 0x1.100000cc091fdp-3, 0x1.119c544dfb2f7p-3,
        0x1.df0bb067d8e18p+2, 0x1.180000d9936a1p-3, 0x1.19c1fb9085a4bp-3, 0x1.d131a71ce3629p+2,
        0x1.1ffffed5f3188p-3, 0x1.21e9dee74bf2dp-3, 0x1.c41b6ff0cd655p+2, 0x1.2800005f06a51p-3,
        0x1.2a141550b313fp-3, 0x1.b7b921702e6d2p+2, 0x1.2fffffe3d339ep-3, 0x1.3240aa75f76dfp-3,
        0x1.abfc8fcb6409dp+2, 0x1.37fffed888d6fp-3, 0x1.3a6fb13cc5db7p-3, 0x1.a0d8f8ed5320dp+2,
        0x1.4000002ca050dp-3, 0x1.42a13e25776bbp-3, 0x1.9642dfa84c2bcp+2, 0x1.47ffff2531f5cp-3,
        0x1.4ad55deb73404p-3, 0x1.8c2fef86e9035p+2, 0x1.5000013774992p-3, 0x1.530c27d0ee307p-3,
        0x1.8296c370caf35p+2, 0x1.57fffedddcb2dp-3, 0x1.5b45a5db4347dp-3, 0x1.796ee52190c0ep+2,
        0x1.5fffff41949a0p-3, 0x1.6381f13bf986ap-3, 0x1.70b092d2255fdp+2, 0x1.67ffff834d3a1p-3,
        0x1.6bc118ec85952p-3, 0x1.6854c62cf2268p+2, 0x1.6fffffd9106eap-3, 0x1.74032f298f6f7p-3,
        0x1.605511f4f84a9p+2, 0x1.780000f649a4fp-3, 0x1.7c4846ca53abcp-3, 0x1.58ab94809d175p+2,
        0x1.7ffffef06bbd8p-3, 0x1.84906df7d76adp-3, 0x1.5152edd2b30a6p+2, 0x1.88000021c6334p-3,
        0x1.8cdbbd965f986p-3, 0x1.4a46251b74296p+2, 0x1.8fffff38d0756p-3, 0x1.952a428e173c7p-3,
        0x1.4380b17b59ebdp+2, 0x1.98000104efca1p-3, 0x1.9d7c14644d23cp-3, 0x1.3cfe5cb1eabd5p+2,
        0x1.a000009417b30p-3, 0x1.a5d14096d76aap-3, 0x1.36bb4b3723db0p+2, 0x1.a7fffeb1c23ecp-3,
        0x1.ae29dab60288dp-3, 0x1.30b3e783071d7p+2, 0x1.afffffb171c13p-3, 0x1.b685fa221a96bp-3,
        0x1.2ae4dd8c0747dp+2, 0x1.b80000aba44bep-3, 0x1.bee5aecdf241fp-3, 0x1.254b1c6fad63bp+2,
        0x1.c0000113d2a3ep-3, 0x1.c7490b3e92543p-3, 0x1.1fe3c9a62c035p+2, 0x1.c7ffff49e00cep-3,
        0x1.cfb020f59eab0p-3, 0x1.1aac3e956a631p+2, 0x1.d000005f611abp-3, 0x1.d81b089e6844ep-3,
        0x1.15a1ff391268dp+2, 0x1.d800014764cebp-3, 0x1.e089d27debf0dp-3, 0x1.10c2bfbc84740p+2,
        0x1.e000014bcea76p-3, 0x1.e8fc916dbc820p-3, 0x1.0c0c5a00ca48ep+2, 0x1.e7ffffd7995bdp-3,
        0x1.f173588b50424p-3, 0x1.077ccbe02169ap+2, 0x1.f00000cc35436p-3, 0x1.f9ee3fd21a40bp-3,
        0x1.031231ee7ffe8p+2, 0x1.f800001d1da33p-3, 0x1.0136ab7dbe15cp-2, 0x1.fd95977d559e5p+1,
        0x1.00000060c13b2p-2, 0x1.05785aaad4f18p-2, 0x1.f549e2675d182p+1, 0x1.0400003885492p-2,
        0x1.09bc3660542d7p-2, 0x1.ed3e2df3f5fecp+1, 0x1.08000052f5a13p-2, 0x1.0e0249a195045p-2,
        0x1.e56f8fa68f2c8p+1, 0x1.0c00002ba1af5p-2, 0x1.124a9e2e7f24bp-2, 0x1.dddb4bffe633fp+1,
        0x1.0fffff76642c1p-2, 0x1.16953e162ffe6p-2, 0x1.d67ed0311d5d5p+1, 0x1.14000033602f0p-2,
        0x1.1ae235f49508ep-2, 0x1.cf57ab8708266p+1, 0x1.17ffffefd1a13p-2, 0x1.1f318db2a9ba1p-2,
        0x1.c86398d11009ep+1, 0x1.1bffff718365dp-2, 0x1.238350c41e3acp-2, 0x1.c1a06e02be47cp+1,
        0x1.1ffffff001e00p-2, 0x1.27d78b2f6395ep-2, 0x1.bb0c1f2fe9a85p+1, 0x1.23ffffe425a6ap-2,
        0x1.2c2e4618faabep-2, 0x1.b4a4c190b18dfp+1, 0x1.28000059ec1dbp-2, 0x1.30878d8583884p-2,
        0x1.ae6880cd82bc2p+1, 0x1.2c000000dd081p-2, 0x1.34e36affdb6d8p-2, 0x1.a855a5270fc15p+1,
        0x1.2fffffc1dcd2bp-2, 0x1.3941ea95875bcp-2, 0x1.a26a8aa9502b6p+1, 0x1.33ffff6c0d4a0p-2,
        0x1.3da31739845f5p-2, 0x1.9ca5a4d2573a0p+1, 0x1.3800006b16793p-2, 0x1.4206fdbc088f0p-2,
        0x1.970579344e33ap+1, 0x1.3c00007358fa3p-2, 0x1.466da6f23311dp-2, 0x1.9188a5aa612eap+1,
        0x1.3fffffc3b18cfp-2, 0x1.4ad71e9282e6bp-2, 0x1.8c2dd641e643dp+1, 0x1.44000000dd2a8p-2,
        0x1.4f4371fa3f2d1p-2, 0x1.86f3c6072f821p+1, 0x1.47ffffbb43713p-2, 0x1.53b2ab3ac333cp-2,
        0x1.81d943da56692p+1, 0x1.4bffffb113bf4p-2, 0x1.5824d6e8ed9c2p-2, 0x1.7cdd2a8add00fp+1,
        0x1.4fffff7f087c9p-2, 0x1.5c9a007446496p-2, 0x1.77fe6444588ebp+1, 0x1.54000088b3814p-2,
        0x1.61123564125f9p-2, 0x1.733be6281a765p+1, 0x1.57ffff7d55966p-2, 0x1.658d7e194a5d5p-2,
        0x1.6e94b73b47d1fp+1, 0x1.5c00008bf2490p-2, 0x1.6a0beb775b28dp-2, 0x1.6a07e15b6ec28p+1,
        0x1.6000009fa853fp-2, 0x1.6e8d865a66cfdp-2, 0x1.659481c701269p+1, 0x1.6400007595fcap-2,
        0x1.73125c0556a7cp-2, 0x1.6139bbaae9d02p+1, 0x1.68000031687dap-2, 0x1.779a792e2cfd0p-2,
        0x1.5cf6bcae0882bp+1, 0x1.6bffff5c8cfe2p-2, 0x1.7c25e9fb452edp-2, 0x1.58cabc561f1cdp+1,
        0x1.6fffff81df231p-2, 0x1.80b4bcfb4dab5p-2, 0x1.54b4f8d3ca5d3p+1, 0x1.73ffffa71385ep-2,
        0x1.8546fe007a9b6p-2, 0x1.50b4bb3b22176p+1, 0x1.78000024a9c2bp-2, 0x1.89dcba7fcf5cfp-2,
        0x1.4cc953159cbe1p+1, 0x1.7c00002eb62b8p-2, 0x1.8e75fec0ba5cfp-2, 0x1.48f218731eeeap+1,
        0x1.80000054a52d1p-2, 0x1.9312d8bb822e9p-2, 0x1.452e69170a729p+1, 0x1.83ffff93a00a3p-2,
        0x1.97b354bb9ad2ap-2, 0x1.417daae924e7fp+1, 0x1.87ffffbdc91c1p-2, 0x1.9c5782771b601p-2,
        0x1.3ddf478855799p+1, 0x1.8bffff6d215e6p-2, 0x1.a0ff6e0ea4a0bp-2, 0x1.3a52b189a0989p+1,
        0x1.9000002f734efp-2, 0x1.a5ab2736bf579p-2, 0x1.36d75e9244ca6p+1, 0x1.9400004eef8b4p-2,
        0x1.aa5ab9fb6e3d0p-2, 0x1.336ccc9089cb7p+1, 0x1.98000041ec76ap-2, 0x1.af0e35176c7e4p-2,
        0x1.30127cb0b9506p+1, 0x1.9c00008044e47p-2, 0x1.b3c5a77071224p-2, 0x1.2cc7f50d75ec7p+1,
        0x1.a000001f8235bp-2, 0x1.b8811e725782ep-2, 0x1.298cc18fbfb37p+1, 0x1.a3ffffb8c6f08p-2,
        0x1.bd40a97b086f3p-2, 0x1.26607154de04bp+1, 0x1.a7ffffb3d63e1p-2, 0x1.c20457d9a3b8ap-2,
        0x1.234297e60bfbbp+1, 0x1.abffff5f45c48p-2, 0x1.c6cc37b8d45e6p-2, 0x1.2032cdb1ace69p+1,
        0x1.b00000508b34cp-2, 0x1.cb985a27e8d37p-2, 0x1.1d30ad4459a2bp+1, 0x1.b40000a84710cp-2,
        0x1.d068cc3e50155p-2, 0x1.1a3bd775034ddp+1, 0x1.b7ffff692e9d8p-2, 0x1.d53d9c49d6627p-2,
        0x1.1753efe18066ap+1, 0x1.bc0000280f14dp-2, 0x1.da16de4e81013p-2, 0x1.14789a38ea052p+1,
        0x1.c0000001121d1p-2, 0x1.def49eac018f0p-2, 0x1.11a9820b8be0cp+1, 0x1.c3ffffef662aap-2,
        0x1.e3d6eea0c7070p-2, 0x1.0ee6532f46ccdp+1, 0x1.c800009845818p-2, 0x1.e8bddf36a8b1bp-2,
        0x1.0c2ebcac73476p+1, 0x1.cbffffb0493bfp-2, 0x1.eda97e0c60d10p-2, 0x1.09827251c7836p+1,
        0x1.cfffffd52961fp-2, 0x1.f299defb3e44bp-2, 0x1.06e1274e459f5p+1, 0x1.d3ffffe2319a4p-2,
        0x1.f78f117139490p-2, 0x1.044a93e737e94p+1, 0x1.d7ffffa4de596p-2, 0x1.fc892638f49e8p-2,
        0x1.01be7231057a5p+1, 0x1.dbffffe729a69p-2, 0x1.00c417c6ab019p-1, 0x1.fe78fbf612660p+0,
        0x1.e000009d66802p-2, 0x1.03461f6b883cfp-1, 0x1.f988ebc05a87cp+0, 0x1.e3ffffb211657p-2,
        0x1.05cab191db8e8p-1, 0x1.f4ac37bcfe6bep+0, 0x1.e80000a3f068ap-2, 0x1.0851d95fb54f0p-1,
        0x1.efe26144ca408p+0, 0x1.ec00001adb060p-2, 0x1.0adb9dc7b54f9p-1, 0x1.eb2af5ebe52a7p+0,
        0x1.effffff5c0d01p-2, 0x1.0d68092550a8dp-1, 0x1.e68580d71fdf0p+0, 0x1.f3ffff93d5fccp-2,
        0x1.0ff7245cb4374p-1, 0x1.e1f193cce5040p+0, 0x1.f7ffffa0dd18fp-2, 0x1.1288f944dd508p-1,
        0x1.dd6ec298b874dp+0, 0x1.fbffffd3a9f1ap-2, 0x1.151d913750f3ep-1, 0x1.d8fca5806a27ep+0,
        0x1.ffffffc481400p-2, 0x1.17b4f598944cap-1, 0x1.d49ad82532170p+0, 0x1.01fffff53786cp-1,
        0x1.1a4f307d83d47p-1, 0x1.d048f851bffebp+0, 0x1.03ffffee301b7p-1, 0x1.1cec4b8a6a382p-1,
        0x1.cc06a7c519db6p+0, 0x1.06000006e36bfp-1, 0x1.1f8c5114eb8bep-1, 0x1.c7d38a34d6786p+0,
        0x1.07ffffd60aa43p-1, 0x1.222f4abeccecbp-1, 0x1.c3af477342ac4p+0, 0x1.0a000037762e8p-1,
        0x1.24d543f99efe8p-1, 0x1.bf99875f54fabp+0, 0x1.0bfffff1c6921p-1, 0x1.277e4598e35d0p-1,
        0x1.bb91f8addd186p+0, 0x1.0dfffff07154bp-1, 0x1.2a2a5b6bc3986p-1, 0x1.b79848301646dp+0,
        0x1.1000002fcdda4p-1, 0x1.2cd9902a59f1ep-1, 0x1.b3ac2705219bfp+0, 0x1.1200001ce1140p-1,
        0x1.2f8bedf0a67c2p-1, 0x1.afcd49ab8ae2ap+0, 0x1.1400004d0f355p-1, 0x1.3241808fcc7bfp-1,
        0x1.abfb6497b9a36p+0, 0x1.1600000fb0c8ap-1, 0x1.34fa52471618bp-1, 0x1.a83630d26d117p+0,
        0x1.18000026f10b3p-1, 0x1.37b66f7579056p-1, 0x1.a47d66b4cf4b1p+0, 0x1.19ffffd0978f8p-1,
        0x1.3a75e290cc78cp-1, 0x1.a0d0c36c21315p+0, 0x1.1bffffd94840bp-1, 0x1.3d38b85b4e4a4p-1,
        0x1.9d30032f2ecefp+0, 0x1.1dffffbda1ea1p-1, 0x1.3ffefbf3cee2fp-1, 0x1.999ae6770fed8p+0,
        0x1.1fffffc989533p-1, 0x1.42c8b9c27900cp-1, 0x1.96112e0d9f1acp+0, 0x1.22000012b8d26p-1,
        0x1.4595fe11975cap-1, 0x1.92929cdaa4e80p+0, 0x1.2400004f4d6afp-1, 0x1.4866d4d224131p-1,
        0x1.8f1ef815c34e8p+0, 0x1.25ffffcc07bdap-1, 0x1.4b3b498b7d010p-1, 0x1.8bb6073e7ffa1p+0,
        0x1.28000042d9639p-1, 0x1.4e136b64540d1p-1, 0x1.88578f4374938p+0, 0x1.2a00003be29a0p-1,
        0x1.50ef452bffd96p-1, 0x1.85035c0042c06p+0, 0x1.2bffffaa91f12p-1, 0x1.53cee3e2f4e0dp-1,
        0x1.81b938542df07p+0, 0x1.2dffffe884891p-1, 0x1.56b256c1a2470p-1, 0x1.7e78ee422ea70p+0,
        0x1.2fffffe87152bp-1, 0x1.5999a9beaaaa1p-1, 0x1.7b424d18fe9b3p+0, 0x1.31fffffc1a721p-1,
        0x1.5c84eafe0e564p-1, 0x1.78152338db8d4p+0, 0x1.33fffff70a372p-1, 0x1.5f74282191d64p-1,
        0x1.74f141122bcaep+0, 0x1.360000436e836p-1, 0x1.62676fde6ccffp-1, 0x1.71d677644252cp+0,
        0x1.37ffffcbe9641p-1, 0x1.655ecee9ffdafp-1, 0x1.6ec49a6fc0515p+0, 0x1.39fffffb6d6cap-1,
        0x1.685a55e67a1e1p-1, 0x1.6bbb7bc2ae969p+0, 0x1.3c00004934fecp-1, 0x1.6b5a12cc07d75p-1,
        0x1.68baf10b02ef8p+0, 0x1.3e00003f5cf7fp-1, 0x1.6e5e13e59def6p-1, 0x1.65c2d0e61500fp+0,
        0x1.4000005280ad9p-1, 0x1.7166691ab4c3ep-1, 0x1.62d2f19f01c90p+0, 0x1.42000049efb65p-1,
        0x1.7473218af3b9dp-1, 0x1.5feb2b86465e4p+0, 0x1.440000035ccb6p-1, 0x1.77844cb4ff1e5p-1,
        0x1.5d0b57c455428p+0, 0x1.4600003346717p-1, 0x1.7a99fba258778p-1, 0x1.5a334f4392254p+0,
        0x1.4800003002575p-1, 0x1.7db43d836768fp-1, 0x1.5762edcf97e0cp+0, 0x1.4a000055bf381p-1,
        0x1.80d3235edeefap-1, 0x1.549a0ea46e31fp+0, 0x1.4c00004ce10e3p-1, 0x1.83f6bd67dc1a8p-1,
        0x1.51d88ed82bcc4p+0, 0x1.4e000016c60e1p-1, 0x1.871f1ca0aaf31p-1, 0x1.4f1e4bdacbf16p+0,
        0x1.4ffffff966f0ap-1, 0x1.8a4c52bff2daep-1, 0x1.4c6b23917657ep+0, 0x1.520000076e6ebp-1,
        0x1.8d7e7175651e8p-1, 0x1.49bef4f459b05p+0, 0x1.5400003d12d3bp-1, 0x1.90b58a976ed56p-1,
        0x1.47199fdf24af4p+0, 0x1.55ffffce7fa8dp-1, 0x1.93f1af03a3a09p-1, 0x1.447b05f13234bp+0,
        0x1.580000331b46ap-1, 0x1.9732f38e83134p-1, 0x1.41e3068d8b41bp+0, 0x1.59ffffc14848ep-1,
        0x1.9a7968471b489p-1, 0x1.3f5185de3aa73p+0, 0x1.5bffffb7cd395p-1, 0x1.9dc5224a8b955p-1,
        0x1.3cc664f8fff15p+0, 0x1.5e000000dcc40p-1, 0x1.a11634df5b93ep-1, 0x1.3a41875853228p+0,
        0x1.6000002efd2fcp-1, 0x1.a46cb30d16323p-1, 0x1.37c2d187962aep+0, 0x1.61ffffeb8088ap-1,
        0x1.a7c8b053920c0p-1, 0x1.354a2891769a9p+0, 0x1.6400000f3ca06p-1, 0x1.ab2a428a1911ap-1,
        0x1.32d770a6f0a4ap+0, 0x1.65ffffe9ec2f4p-1, 0x1.ae917d4ce7239p-1, 0x1.306a90751a948p+0,
        0x1.6800003d43966p-1, 0x1.b1fe7708b998ap-1, 0x1.2e036d7a153c7p+0, 0x1.69ffffab67783p-1,
        0x1.b57142e575464p-1, 0x1.2ba1f05006cb6p+0, 0x1.6bffffcb65f89p-1, 0x1.b8e9f981efd2fp-1,
        0x1.2945fe948d9f7p+0, 0x1.6dffffc5609a9p-1, 0x1.bc68afaed6ff1p-1, 0x1.26ef81533411ep+0,
        0x1.6fffffc4eef86p-1, 0x1.bfed7c62205fep-1, 0x1.249e60e70978cp+0, 0x1.72000004d98b3p-1,
        0x1.c3787716968adp-1, 0x1.2252861be7751p+0, 0x1.73ffffc155075p-1, 0x1.c709b5ec6fd4ep-1,
        0x1.200bdb5d53311p+0, 0x1.760000498c203p-1, 0x1.caa15323d08c1p-1, 0x1.1dca493433f65p+0,
        0x1.77fffff1e5636p-1, 0x1.ce3f64147c12cp-1, 0x1.1b8dbbfe294a8p+0, 0x1.7a0000226d45ap-1,
        0x1.d1e404120eb7fp-1, 0x1.19561d15f8278p+0, 0x1.7c0000250a5aap-1, 0x1.d58f4b112a1e1p-1,
        0x1.172358a59d565p+0, 0x1.7e0000482f82ep-1, 0x1.d94153576bdf0p-1, 0x1.14f55a22a1c5bp+0,
        0x1.800000205003ep-1, 0x1.dcfa364d69ff7p-1, 0x1.12cc0e37eb26fp+0, 0x1.81ffffbf99411p-1,
        0x1.e0ba0ebf98f51p-1, 0x1.10a7616ddd5d6p+0, 0x1.840000248e3a3p-1, 0x1.e480f9bb7f565p-1,
        0x1.0e873fb84e05cp+0, 0x1.860000145c157p-1, 0x1.e84f10a10b3abp-1, 0x1.0c6b97cbd7b1ep+0,
        0x1.88000022631b9p-1, 0x1.ec2470be1f047p-1, 0x1.0a5456d0b3ee6p+0, 0x1.8a000045f7828p-1,
        0x1.f00136c45ba1cp-1, 0x1.08416af2a0f09p+0, 0x1.8bfffffc686cfp-1, 0x1.f3e57f03c824bp-1,
        0x1.0632c33502220p+0, 0x1.8dffffd8644c6p-1, 0x1.f7d168774261dp-1, 0x1.04284dd5b3019p+0,
        0x1.8fffffe4e7937p-1, 0x1.fbc511a99a641p-1, 0x1.0221f9f69840bp+0, 0x1.920000435251fp-1,
        0x1.ffc099eb22390p-1, 0x1.001fb6f7c51e8p+0,
    });

constexpr double TAN_SERIES3 = 0x1.5555555555555p-2;
constexpr double TAN_SERIES5 = 0x1.11111111107c6p-3;
constexpr double TAN_SERIES7 = 0x1.ba1ba1cdb8745p-5;
constexpr double TAN_SERIES9 = 0x1.664ed49cfc666p-6;
constexpr double TAN_SERIES11 = 0x1.2385a3cf2e4eap-7;
constexpr double TAN_TABLE_SERIES3 = 0x1.5555555554dbdp-2;
constexpr double TAN_TABLE_SERIES5 = 0x1.11112e0a6b45fp-3;
constexpr double TAN_TABLE_INDEX_OFFSET = -15.5;
constexpr double TAN_IDENTITY_LIMIT = 0x1.b096cp-27;
constexpr double TAN_SERIES_LIMIT = 0x1.f212dp-5;
constexpr double TAN_TABLE_LIMIT = 0x1.92f1ap-1;
constexpr double TAN_SHORT_REDUCTION_LIMIT = 25.0;
constexpr double TAN_REDUCTION_LIMIT = 0x1.7d784p+26;
// The third part of half pi in full, for the reduction of arguments up to 25.
constexpr double TAN_HALF_PI_PART3 = -0x1.cb3b399d747f2p-55;

// The series of tan in odd powers, after the first term and divided by the cube.
CUJU_HOST_DEVICE double TanSeries(double square) {
    return fma(
        square,
        fma(square, fma(square, fma(square, TAN_SERIES11, TAN_SERIES9), TAN_SERIES7), TAN_SERIES5),
        TAN_SERIES3);
}

// s_tan.c's algorithm i for arguments up to 25, which returns the parity of the multiple of half
// pi and leaves the remainder in head and tail.
CUJU_HOST_DEVICE int TanReduceShort(double argument, double *head, double *tail) {
    double shifted = fma(argument, SIN_COS_INVERSE_HALF_PI, SIN_COS_ROUND_SHIFT);
    double multiple = shifted - SIN_COS_ROUND_SHIFT;
    double remainder =
        fma(-multiple, SIN_COS_HALF_PI_PART2, fma(-multiple, SIN_COS_HALF_PI_PART1, argument));
    // The product with the third part feeds two subtractions, and the FMA build fuses it into both.
    double reduced = fma(-multiple, TAN_HALF_PI_PART3, remainder);
    *head = reduced;
    *tail = fma(-multiple, TAN_HALF_PI_PART3, remainder - reduced);
    return (int)(uint32_t)DoubleBits(shifted) & 1;
}

// The tangent of head plus tail when `odd` is 0 and minus the cotangent otherwise, by the series,
// for a sum up to 0.0608 in magnitude.
CUJU_HOST_DEVICE double TanFromSeries(double head, double tail, int odd) {
    double square = head * head;
    double correction = fma(head * square, TanSeries(square), tail);
    double sum = head + correction;
    if (!odd) return sum;
    double sumTail =
        (fabs(head) > fabs(correction)) ? (head - sum) + correction : (correction - sum) + head;
    // dla.h's `DIV2` of 1 by the sum and its tail, whose `MUL12` is exact with `fma`.
    double quotient = 1.0 / sum;
    double product = quotient * sum;
    double productTail = fma(quotient, sum, -product);
    double quotientTail = fma(-sumTail, quotient, ((1.0 - product) - productTail) + 0.0) / sum;
    double result = quotient + quotientTail;
    double resultTail = (quotient - result) + quotientTail;
    return -(resultTail + result);
}

// The tangent of a sum when `odd` is 0 and minus the cotangent otherwise, from the table, with
// `magnitude` and `tail` the magnitude of the sum from 0.0608 to 0.787 and `sign` its sign.
CUJU_HOST_DEVICE double TanFromTable(double magnitude, double tail, double sign, int odd) {
    const double *table = CUJU_TABLE(TAN_TABLE);
    int index = 3 * (int)fma(magnitude, 256.0, TAN_TABLE_INDEX_OFFSET);
    double offset = (magnitude - table[index]) + tail;
    double square = offset * offset;
    double polynomial =
        fma(offset * square, fma(square, TAN_TABLE_SERIES5, TAN_TABLE_SERIES3), offset);
    double tangent = table[index + 1], cotangent = table[index + 2];
    if (odd) {
        double quotient = polynomial * (tangent + cotangent) / (tangent + polynomial);
        return -sign * (cotangent - quotient);
    }
    double quotient = polynomial * (cotangent + tangent) / (cotangent - polynomial);
    return sign * (tangent + quotient);
}

CUJU_HOST_DEVICE_NO_INLINE double tan(double argument) {
    if ((DoubleBits(argument) >> 52 & 0x7ff) == 0x7ff) return argument - argument;
    double magnitude = (argument < 0.0) ? -argument : argument;
    if (magnitude <= TAN_IDENTITY_LIMIT) return argument;
    if (magnitude <= TAN_SERIES_LIMIT) {
        double square = argument * argument;
        return fma(argument * square, TanSeries(square), argument);
    }
    if (magnitude <= TAN_TABLE_LIMIT)
        return TanFromTable(magnitude, 0.0, (argument < 0.0) ? -1.0 : 1.0, 0);
    double head, tail;
    int odd;
    if (magnitude <= TAN_SHORT_REDUCTION_LIMIT) {
        odd = TanReduceShort(argument, &head, &tail);
    } else {
        // s_tan.c's algorithm ii is the reduction of sin and cos, and algorithm iii is `__branred`.
        if (magnitude <= TAN_REDUCTION_LIMIT)
            odd = ReduceHalfPi(argument, &head, &tail) & 1;
        else
            odd = ReduceHalfPiLarge(argument, &head, &tail) & 1;
        double sum = head + tail;
        tail = (fabs(head) > fabs(tail)) ? (head - sum) + tail : (tail - sum) + head;
        head = sum;
    }
    if (fabs(head) <= TAN_SERIES_LIMIT) return TanFromSeries(head, tail, odd);
    if (head < 0.0) return TanFromTable(-head, -tail, -1.0, odd);
    return TanFromTable(head, tail, 1.0, odd);
}

}  // namespace libm
}  // namespace cuju
