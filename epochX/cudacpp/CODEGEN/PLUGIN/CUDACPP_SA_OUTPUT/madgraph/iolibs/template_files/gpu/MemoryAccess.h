// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created by: D. Massaro (Feb 2026) for the MG5aMC CUDACPP plugin.

#ifndef MemoryAccess_H
#define MemoryAccess_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
class EventVariableRef {
  const fptype* const ptr;

 public:
  __host__ __device__ explicit EventVariableRef(const fptype* ptr_): ptr(ptr_) {}

  using sv_type = const fptype_sv;
  __host__ __device__ operator sv_type&() const {
    return *reinterpret_cast<sv_type*>(ptr);
  }
  __host__ __device__ const fptype* data() const {
    return ptr;
  }
};

// A smaller view to be used together with AoSoAView to describe particle
// momenta according to AOSOA[npagM][npar][np4][neppM] where nevt=npagM*neppM
struct ParticleView {
  static constexpr int singleDim = CPPProcess::np4;
  static constexpr int eventDim = CPPProcess::npar * singleDim;

  EventVariableRef en, px, py, pz;

  __host__ __device__
  ParticleView(const fptype* ptr_, int stride)
    : en(ptr_)
    , px(ptr_ + stride)
    , py(ptr_ + 2*stride)
    , pz(ptr_ + 3*stride)
  {}
};

// A view class over an AoSoA internal layout an AOSOA[npagM][inner view dimensions][neppM] where nevt=npagM*neppM
template<class InnerView>
class AoSoAView {
  const fptype* ptr;
#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
  // -----------------------------------------------------------------------------------------------
  // --- GPUs: neppM is best set to a power of 2 times the number of fptype's in a 32-byte cacheline
  // --- This is relevant to ensure coalesced access to momenta in global memory
  // --- Note that neppR is hardcoded and may differ from neppM and neppV on some platforms
  // -----------------------------------------------------------------------------------------------
  static constexpr int neppM = 32/sizeof(fptype);
#else
  // -----------------------------------------------------------------------------------------------
  // --- CPUs: neppM is best set equal to the number of fptype's (neppV) in a vector register
  // --- This is relevant to ensure faster access to momenta from C++ memory cache lines
  // --- However, neppM is now decoupled from neppV (issue #176) and can be separately hardcoded
  // --- In practice, neppR, neppM and neppV could now (in principle) all be different
  // -----------------------------------------------------------------------------------------------
#ifdef MGONGPU_CPPSIMD
  static constexpr int neppM = MGONGPU_CPPSIMD;
#else
  static constexpr int neppM = 1; // (DEFAULT) neppM=neppV for optimal performance (NB: this is equivalent to AOS)
#endif
#endif /* clang-format on */

  // SANITY CHECK: check that neppM is a power of two
  static_assert( ispoweroftwo( neppM ), "neppM is not a power of 2" );

  __host__ __device__ int pageStride(const int ievt) const {
    const int iPage = ievt / neppM;
    const int pageStride = iPage * InnerView::eventDim * neppM;
    return pageStride;
  }

 public:
  __host__ __device__ AoSoAView(const fptype* ptr_): ptr(ptr_) {}
  __host__ __device__ AoSoAView page_at(const int ievt) {
    const int pageStride = this->pageStride(ievt);
    return AoSoAView{ptr + pageStride};
  }
  __host__ __device__ InnerView operator[](const int i) const {
#ifdef MGONGPUCPP_GPUIMPL
    // the event number is given by the thread index in the grid
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
    // we first need to select the page, shifting by many page dimension
    const int pageStride = this->pageStride(ievt);
    // then, within the page, we need to shift by many InnerView dimensions
    const int innerStride = i * InnerView::singleDim * neppM;
    // finally, we shift by the vector lane
    const int iLane = ievt % neppM;
    const int stride = pageStride + innerStride + iLane;
#else
    // we are already in the correct memory page, pointing to the zero-th event
    // we shift by a certain number of particles
    const int innerStride = i * InnerView::singleDim * neppM;
    const int stride = innerStride;
#endif
    return InnerView{ptr + stride, neppM};
  }
};

using MomentaView = AoSoAView<ParticleView>;
} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccess_H
