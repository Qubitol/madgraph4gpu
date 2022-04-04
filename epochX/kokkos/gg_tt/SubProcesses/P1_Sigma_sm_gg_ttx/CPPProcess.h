//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.9.7, 2021-11-29
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================


#include "../../src/HelAmps_sm.h"

#ifndef MG5_Sigma_sm_gg_ttx_H
#define MG5_Sigma_sm_gg_ttx_H

#include <vector>
#include <cassert>
#include <iostream>

#include "mgOnGpuConfig.h"
#include "mgOnGpuVectors.h"
#include "mgOnGpuTypes.h"

#include "Kokkos_Core.hpp"

#include "Parameters_sm.h"

namespace mg5amcGpu
{

///////////////////////// -*- C++ -*- /////////////////////////////

//==========================================================================
// A class for calculating the matrix elements for
// Process: g g > t t~ WEIGHTED<=2 @1
//--------------------------------------------------------------------------

template <class ExecSpace>
class CPPProcess
{
public:

  CPPProcess(int numiterations, int leaguesize, int teamsize,
             bool verbose = false, bool debug = false);


  ~CPPProcess() = default;

  // Initialize process.
	virtual void initProc(const std::string& param_card_name);
  const std::string getCompiler();

  virtual int code() const { return 1; }

  const typename Kokkos::View<fptype*,ExecSpace>::HostMirror &getMasses() const { return hmME; }

  void setInitial(int inid1, int inid2) {
    id1 = inid1;
    id2 = inid2;
  }

  int getDim() const { return dim; }

  int getNIOParticles() const { return mgOnGpu::npar; }


  // Constants for array limits
  static const int ninitial = mgOnGpu::npari;
  static const int nexternal = mgOnGpu::npar;
  static const int nprocesses = 1;
  static const int ncomb = mgOnGpu::ncomb;

  Kokkos::View<int**,ExecSpace> cHel;
  typename Kokkos::View<int**,ExecSpace>::HostMirror hHel;

  Kokkos::View<fptype*,ExecSpace> cmME;
  typename Kokkos::View<fptype*,ExecSpace>::HostMirror hmME;

  Kokkos::View<fptype*,ExecSpace> cIPC;
  typename Kokkos::View<fptype*,ExecSpace>::HostMirror hIPC;

  Kokkos::View<fptype*,ExecSpace> cIPD;
  typename Kokkos::View<fptype*,ExecSpace>::HostMirror hIPD;

  const short* get_tHel_ptr() const;

  cxtype* get_tIPC_ptr();
  const cxtype* get_tIPC_ptr() const;

  fptype* get_tIPD_ptr();
  const fptype* get_tIPD_ptr() const;
private:
  int m_numiterations;
  // gpu variables
  int league_size;
  int team_size;
  int dim; // league_size * team_size;

  // print verbose info
  bool m_verbose;

  // print debug info
  bool m_debug;

  static const int nwavefuncs = 6;
  static const int namplitudes = 3;
  static const int wrows = 7;
  //static const int nioparticles = 4;

  // Pointer to the model parameters
  Parameters_sm *pars;

  // Initial particle ids
  int id1, id2;

};

// Constants for array limits
template <typename ExecSpace> const int CPPProcess<ExecSpace>::ninitial;
template <typename ExecSpace> const int CPPProcess<ExecSpace>::nprocesses;
template <typename ExecSpace> const int CPPProcess<ExecSpace>::ncomb;
template <typename ExecSpace> const int CPPProcess<ExecSpace>::nexternal;



///////////////////////// -*- C++ -*- /////////////////////////////
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g g > t t~ WEIGHTED<=2 @1

using mgOnGpu::ncomb; // number of helicity combinations
//--------------------------------------------------------------------------

// Evaluate |M|^2 for each subprocess
// NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event



template <typename hel_t, typename mom_t, typename ipd_t, typename ipc_t>
KOKKOS_FUNCTION void calculate_wavefunctions(
  const mom_t& allmomenta,
  const hel_t& cHel,
  const ipd_t& cIPD,
  const ipc_t& cIPC,
  fptype_sv& allMEs
  )
{
  using namespace MG5_sm;
  // The number of colors
  constexpr int ncolor = 2;

  // Local TEMPORARY variables for a subset of Feynman diagrams in the given CUDA event (ievt) or C++ event page (ipagV)
  // [NB these variables are reused several times (and re-initialised each time) within the same event or event page]
  cxtype_sv w_sv[mgOnGpu::nwf][mgOnGpu::nw6]; // particle wavefunctions within Feynman diagrams (nw6 is often 6, the dimension of spin 1/2 or spin 1 particles)
  cxtype_sv amp_sv[1]; // invariant amplitude for one given Feynman diagram

  // Local variables for the given CUDA event (ievt) or C++ event page (ipagV)
  cxtype_sv jamp_sv[ncolor] = {}; // sum of the invariant amplitudes for all Feynman diagrams in the event or event page

  // Calculate wavefunctions for all processes
      // SYCL kernels take an input buffer with momenta for all events
      // const fptype* momenta = allmomenta;
      // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
      for( int i=0; i<ncolor; i++ ){ jamp_sv[i] = cxzero_sv(); }

      // *** DIAGRAM 1 OF 3 ***

      // Wavefunction(s) for diagram number 1
      vxxxxx(Kokkos::subview(allmomenta, 0, Kokkos::ALL), 0., cHel[0], -1, w_sv[0]);

      vxxxxx(Kokkos::subview(allmomenta, 1, Kokkos::ALL), 0., cHel[1], -1, w_sv[1]);

      oxxxxx(Kokkos::subview(allmomenta, 2, Kokkos::ALL), cIPD[0], cHel[2], +1, w_sv[2]);

      ixxxxx(Kokkos::subview(allmomenta, 3, Kokkos::ALL), cIPD[0], cHel[3], -1, w_sv[3]);

      VVV1P0_1( w_sv[0], w_sv[1], cxmake( cIPC[0], cIPC[1] ), 0., 0., w_sv[4] );

      // Amplitude(s) for diagram number 1
      FFV1_0( w_sv[3], w_sv[2], w_sv[4], cxmake( cIPC[2], cIPC[3] ), &amp_sv[0] );
      jamp_sv[0] += +cxtype(0,1)*amp_sv[0];
      jamp_sv[1] -= cxtype(0,1)*amp_sv[0];

      // *** DIAGRAM 2 OF 3 ***

      // Wavefunction(s) for diagram number 2
      FFV1_1( w_sv[2], w_sv[0], cxmake( cIPC[2], cIPC[3] ), cIPD[0], cIPD[1], w_sv[4] );

      // Amplitude(s) for diagram number 2
      FFV1_0( w_sv[3], w_sv[4], w_sv[1], cxmake( cIPC[2], cIPC[3] ), &amp_sv[0] );
      jamp_sv[0] -= amp_sv[0];

      // *** DIAGRAM 3 OF 3 ***

      // Wavefunction(s) for diagram number 3
      FFV1_2( w_sv[3], w_sv[0], cxmake( cIPC[2], cIPC[3] ), cIPD[0], cIPD[1], w_sv[4] );

      // Amplitude(s) for diagram number 3
      FFV1_0( w_sv[4], w_sv[2], w_sv[1], cxmake( cIPC[2], cIPC[3] ), &amp_sv[0] );
      jamp_sv[1] -= amp_sv[0];

  // *** COLOR ALGEBRA BELOW ***
  // (This method used to be called CPPProcess::matrix_1_gg_ttx()?)

  // The color matrix
      static constexpr fptype denom[ncolor] = {3, 3}; // 1-D array[2]
      static constexpr fptype cf[ncolor][ncolor] = {
        {16, -2},
        {-2, 16}}; // 2-D array[2][2]

  // Sum and square the color flows to get the matrix element
  // (compute |M|^2 by squaring |M|, taking into account colours)
  fptype_sv deltaMEs = { 0 }; // all zeros
  for( int icol = 0; icol < ncolor; icol++ )
  {
    cxtype_sv ztemp_sv = cxzero_sv();
    for( int jcol = 0; jcol < ncolor; jcol++ )
      ztemp_sv += cf[icol][jcol] * jamp_sv[jcol];
    deltaMEs += cxreal( ztemp_sv * cxconj( jamp_sv[icol] ) ) / denom[icol];
  }

  // *** STORE THE RESULTS ***

  // Store the leading color flows for choice of color
  // (NB: jamp2_sv must be an array of fptype_sv)
  // for( int icol = 0; icol < ncolor; icol++ )
  // jamp2_sv[0][icol] += cxreal( jamp_sv[icol]*cxconj( jamp_sv[icol] ) );

  // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event(s)
  // FIXME: assume process.nprocesses == 1 for the moment (eventually: need a loop over processes here?)
  allMEs += deltaMEs;
  //printf( "calculate_wavefunction: %6d %2d %f\n", ievt, ihel, allMEs[ievt] );

    mgDebug( 1, __FUNCTION__ );
    return;
  }


//--------------------------------------------------------------------------

template <typename ExecSpace>
CPPProcess<ExecSpace>::CPPProcess(
    int numiterations, int leaguesize, int teamsize,
    bool verbose, bool debug): 
      m_numiterations(numiterations), league_size(leaguesize), 
      team_size(teamsize), 
      dim(league_size * team_size),
      cHel("cHel",mgOnGpu::ncomb,mgOnGpu::npar), hHel("hHel",mgOnGpu::ncomb,mgOnGpu::npar), 
      cmME("cmME",mgOnGpu::npar), hmME("hmME",mgOnGpu::npar),
      cIPC("cIPC",4), hIPC("hIPC",4), 
      cIPD("cIPD",2), hIPD("hIPD",2)
{

  // Helicities for the process - nodim
  static const int tHel[ncomb][nexternal] = {
      {-1, -1, -1, -1},
      {-1, -1, -1, 1},
      {-1, -1, 1, -1},
      {-1, -1, 1, 1},
      {-1, 1, -1, -1},
      {-1, 1, -1, 1},
      {-1, 1, 1, -1},
      {-1, 1, 1, 1},
      {1, -1, -1, -1},
      {1, -1, -1, 1},
      {1, -1, 1, -1},
      {1, -1, 1, 1},
      {1, 1, -1, -1},
      {1, 1, -1, 1},
      {1, 1, 1, -1},
      {1, 1, 1, 1}};

  for(int i=0;i<mgOnGpu::ncomb;++i)
      for(int j=0;j<mgOnGpu::npar;++j){
          hHel(i,j) = tHel[i][j];
      }
  Kokkos::deep_copy(cHel,hHel);
}

//--------------------------------------------------------------------------

// Initialize process (with parameters read from user cards)
template <typename ExecSpace>
void CPPProcess<ExecSpace>::initProc( const std::string& param_card_name )
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance();
  SLHAReader slha(param_card_name, m_verbose);
  pars->setIndependentParameters(slha);
  pars->setIndependentCouplings();
  if (m_verbose) {
      pars->printIndependentParameters();
    pars->printIndependentCouplings();
  }
  pars->setDependentParameters();
  pars->setDependentCouplings();
  // Set external particle masses for this matrix element
  hmME(0) = ( pars->ZERO );
  hmME(1) = ( pars->ZERO );
  hmME(2) = ( pars->mdl_MT );
  hmME(3) = ( pars->mdl_MT );
  Kokkos::deep_copy(cmME,hmME);
  
  const cxtype tIPC[2] = { cxmake( pars->GC_10 ), cxmake( pars->GC_11 ) };
  for(int i=0;i<2;++i){
    hIPC(i*2 + 0) = tIPC[i].real();
    hIPC(i*2 + 1) = tIPC[i].imag();
  }
  Kokkos::deep_copy(cIPC,hIPC);

    const fptype tIPD[2] = { (fptype)pars->mdl_MT, (fptype)pars->mdl_WT };
  for(int i=0;i<2;++i)
    hIPD(i) = tIPD[i];
  Kokkos::deep_copy(cIPD,hIPD);

}


// Retrieve the compiler that was used to build this module
template<typename ExecSpace>
const std::string CPPProcess<ExecSpace>::getCompiler()
{
  std::stringstream out;
  // CUDA version (NVCC)
#ifdef __CUDACC__
#if defined __CUDACC_VER_MAJOR__ && defined __CUDACC_VER_MINOR__ && defined __CUDACC_VER_BUILD__
  out << "nvcc " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__;
#else
  out << "nvcc UNKNOWN";
#endif
  out << " (";
#endif
  // ICX version (either as CXX or as host compiler inside NVCC)
#if defined __INTEL_COMPILER
#error "icc is no longer supported: please use icx"
#elif defined __INTEL_LLVM_COMPILER // alternative: __INTEL_CLANG_COMPILER
  out << "icx " << __INTEL_LLVM_COMPILER << " (";
#endif
  // CLANG version (either as CXX or as host compiler inside NVCC or inside ICX)
#if defined __clang__
#if defined __clang_major__ && defined __clang_minor__ && defined __clang_patchlevel__
  out << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
  // GCC toolchain version inside CLANG
  std::string tchainout;
  std::string tchaincmd = "readelf -p .comment $(${CXX} -print-libgcc-file-name) |& grep 'GCC: (GNU)' | grep -v Warning | sort -u | awk '{print $5}'";
  std::unique_ptr<FILE, decltype(&pclose)> tchainpipe( popen( tchaincmd.c_str(), "r" ), pclose );
  if ( !tchainpipe ) throw std::runtime_error( "`readelf ...` failed?" );
  std::array<char, 128> tchainbuf;
  while ( fgets( tchainbuf.data(), tchainbuf.size(), tchainpipe.get() ) != nullptr ) tchainout += tchainbuf.data();
  if(tchainout.size() > 0) tchainout.pop_back(); // remove trailing newline
#if defined __CUDACC__ or defined __INTEL_LLVM_COMPILER
  out << ", gcc " << tchainout;
#else
  out << " (gcc " << tchainout << ")";
#endif
#else
  out << "clang UNKNOWKN";
#endif
#else
  // GCC version (either as CXX or as host compiler inside NVCC)
#if defined __GNUC__ && defined __GNUC_MINOR__ && defined __GNUC_PATCHLEVEL__
  out << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
  out << "gcc UNKNOWKN";
#endif
#endif
#if defined __CUDACC__ or defined __INTEL_LLVM_COMPILER
  out << ")";
#endif
  return out.str();
}

//--------------------------------------------------------------------------
// Define pointer accessors
template <typename ExecSpace>
const short* CPPProcess<ExecSpace>::get_tHel_ptr() const {return &(**hHel);}

template <typename ExecSpace>
cxtype* CPPProcess<ExecSpace>::get_tIPC_ptr() {return hIPC;}
template <typename ExecSpace>
const cxtype* CPPProcess<ExecSpace>::get_tIPC_ptr() const {return hIPC;}

template <typename ExecSpace>
fptype* CPPProcess<ExecSpace>::get_tIPD_ptr() {return hIPD;}
template <typename ExecSpace>
const fptype* CPPProcess<ExecSpace>::get_tIPD_ptr() const {return hIPD;}

//--------------------------------------------------------------------------
template <typename mom_t, typename out_t, typename hel_t, typename ipd_t, 
          typename ipc_t, typename igh_t, typename ngh_t>
void sigmaKin_setup(
    const mom_t& momenta,
    out_t& allMEs,
    const hel_t& cHel,
    const ipd_t& cIPD,
    const ipc_t& cIPC,
    igh_t& iGoodHel,
    ngh_t& nGoodHel,
    const int& ncomb,
    const int& league_size,
    const int& team_size) 
{
  Kokkos::View<int*,Kokkos::DefaultExecutionSpace> isGoodHel("isGoodHel",ncomb); // has to be constant index, but should be `ncomb`

  using member_type = typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy( league_size, team_size );
  Kokkos::parallel_for(__func__,policy, 
  KOKKOS_LAMBDA(member_type team_member){
    const int ievt = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
    fptype allMEsLast = 0;

    auto local_mom = Kokkos::subview(momenta,ievt,Kokkos::ALL,Kokkos::ALL);
    for (int ihel = 0;ihel < ncomb;++ihel)
    {
      auto local_cHel = Kokkos::subview(cHel,ihel,Kokkos::ALL);
      calculate_wavefunctions(local_mom, local_cHel, cIPD, cIPC, allMEs[ievt]);
      if (allMEs[ievt] != allMEsLast)
      {
        isGoodHel(ihel) = true;
      }
      allMEsLast = allMEs[ievt];
    }
  });

  Kokkos::parallel_for(__func__,Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1),
  KOKKOS_LAMBDA(const int& i){
    for(int ihel=0;ihel < ncomb;++ihel){

      if(isGoodHel(ihel)){
        iGoodHel(nGoodHel(0)) = ihel;
        nGoodHel(0)++;
      }

    }
  });
}


//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.
template <typename mom_t, typename out_t, typename hel_t, typename ipd_t, 
          typename ipc_t, typename igh_t, typename ngh_t>
void sigmaKin(const mom_t& momenta, out_t& allMEs, const hel_t& cHel,
    const ipd_t& cIPD, const ipc_t& cIPC, const igh_t& iGoodHel,
    const ngh_t& nGoodHel, const int& ncomb, const int& league_size,
    const int& team_size)
{
  using member_type = typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy( league_size, team_size );
  Kokkos::parallel_for(__func__,policy, 
  KOKKOS_LAMBDA(member_type team_member){

    const int ievt = team_member.league_rank() * team_member.team_size() + team_member.team_rank();
    
    // Denominators: spins, colors and identical particles
    constexpr int denominators = 512;
    allMEs[ievt] = 0;

    // PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS
    // (in both CUDA and C++, using precomputed good helicities)
    // FIXME: assume process.nprocesses == 1 for the moment (eventually: need a loop over processes here?)
    auto local_mom = Kokkos::subview(momenta,ievt,Kokkos::ALL,Kokkos::ALL);
    for (int ighel = 0;ighel < nGoodHel(0);++ighel)
    {
      auto local_cHel = Kokkos::subview(cHel,iGoodHel(ighel),Kokkos::ALL);
      calculate_wavefunctions(local_mom, local_cHel, cIPD, cIPC, allMEs[ievt]);
    }
    // PART 2 - FINALISATION (after calculate_wavefunctions)
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    // [NB 'sum over final spins, average over initial spins', eg see
    // https://www.uzh.ch/cmsssl/physik/dam/jcr:2e24b7b1-f4d7-4160-817e-47b13dbf1d7c/Handout_4_2016-UZH.pdf]
    // FIXME: assume process.nprocesses == 1 for the moment (eventually: need a loop over processes here?)
    allMEs[ievt] /= (fptype)denominators;

  });// end parallel for

}


//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------


} // end namespace mg5amcGpu

#endif // MG5_Sigma_sm_gg_ttx_H