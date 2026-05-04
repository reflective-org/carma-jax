!! Diagnostic-instrumented sulfate test (Phase 0).
!!
!! Reads ONE scenario from argv[1], writes per-substep dumps of every
!! microfast/microslow intermediate array into argv[2] (output directory).
!! Optional argv[3]: nstep_max — stop after this many timesteps
!! (default: full simulation). Useful when running across many
!! scenarios where only the first substep is needed.
!! No JSON output — pure binary stream files for the Python differential
!! validation harness.
!!
!! Scenario line (whitespace-separated):
!!   T_K  p_hPa  rh_fraction  h2so4_pptv  aerosol_mu_nm  aerosol_sigma_g
!!
!! Output files (in argv[2] directory):
!!   substep_<NNNN>_pc.bin           shape (NBIN, NELEM)
!!   substep_<NNNN>_gc.bin           shape (NGAS,)
!!   substep_<NNNN>_t.bin            shape (NZ,)
!!   substep_<NNNN>_p.bin            shape (NZ,)
!!   substep_<NNNN>_rhoa.bin         shape (NZ,)
!!   substep_<NNNN>_zmet.bin         shape (NZ,)
!!   substep_<NNNN>_wtpct.bin        shape (NZ,)  -- sulfate wt% from Tabazadeh
!!   substep_<NNNN>_sulfdens.bin     shape (NZ,)  -- sulfate_density(wtpct, t)
!!   substep_<NNNN>_sulfsurf.bin     shape (NZ,)  -- sulfate_surf_tens(wtpct, t)
!!   substep_<NNNN>_pvapl.bin        shape (NZ, NGAS)
!!   substep_<NNNN>_pvapi.bin        shape (NZ, NGAS)
!!   substep_<NNNN>_supsatl.bin      shape (NZ, NGAS)
!!   substep_<NNNN>_supsati.bin      shape (NZ, NGAS)
!!   substep_<NNNN>_akelvin.bin      shape (NZ, NGAS)
!!   substep_<NNNN>_akelvini.bin     shape (NZ, NGAS)
!!   substep_<NNNN>_gro.bin          shape (NZ, NBIN, NGROUP)
!!   substep_<NNNN>_gro1.bin         shape (NZ, NBIN, NGROUP)
!!   substep_<NNNN>_rhompe.bin       shape (NBIN, NELEM)
!!   substep_<NNNN>_rnuclg.bin       shape (NBIN, NGROUP, NGROUP)
!!   substep_<NNNN>_rnucpe.bin       shape (NBIN, NELEM)
!!   substep_<NNNN>_growpe.bin       shape (NBIN, NELEM)
!!   substep_<NNNN>_evappe.bin       shape (NBIN, NELEM)
!!   substep_<NNNN>_growlg.bin       shape (NBIN, NGROUP)
!!   substep_<NNNN>_evaplg.bin       shape (NBIN, NGROUP)
!!   substep_<NNNN>_gasprod.bin      shape (NGAS,)
!!   substep_<NNNN>_rlheat.bin       shape (NZ,)
!!   substep_<NNNN>_pconmax.bin      shape (NZ, NGROUP)
!!   substep_<NNNN>_coaglg.bin       shape (NZ, NBIN, NGROUP)
!!   substep_<NNNN>_coagpe.bin       shape (NZ, NBIN, NELEM)
!!   substep_<NNNN>_ckernel.bin      shape (NZ, NBIN, NBIN, NGROUP, NGROUP) — only step 1 (static)
!!
!! Arrays are written in Fortran column-major order as raw float64 stream.
!! Numpy reader uses np.fromfile + reshape with order='F'.
!!
!! Build: `bash scripts/fortran_patch/apply_diagnostic_patch.sh`.

program carma_sulfatetest_diagnostic
  implicit none
  call test_sulfate_diagnostic()
end program

subroutine test_sulfate_diagnostic()
  use carma_precision_mod
  use carma_constants_mod
  use carma_enums_mod
  use carma_types_mod
  use carmaelement_mod
  use carmagroup_mod
  use carmagas_mod
  use carmastate_mod
  use carma_mod
  use atmosphere_mod
  use sulfate_utils, only: sulfate_density, sulfate_surf_tens

  implicit none

  integer, parameter        :: NZ           = 1
  integer, parameter        :: NZP1         = NZ+1
  integer, parameter        :: NELEM        = 1
  integer, parameter        :: NBIN         = 38
  integer, parameter        :: NGROUP       = 1
  integer, parameter        :: NSOLUTE      = 0
  integer, parameter        :: NGAS         = 2
  integer, parameter        :: NWAVE        = 0
  integer, parameter        :: LUNOPRT      = 6

  real(kind=f), parameter   :: dtime  = 1800._f
  real(kind=f), parameter   :: deltaz = 10000._f
  real(kind=f), parameter   :: zmin   = 17000._f
  integer, parameter        :: nstep  = 180000 / int(dtime)

  integer, parameter        :: I_H2SO4  = 1

  character(len=512)        :: scenario_path, output_dir, nstep_max_str
  real(kind=f)              :: T_scen, p_scen_hPa, rh_scen
  real(kind=f)              :: h2so4_pptv, mu_nm, sigma_g
  integer                   :: unit_in, ios
  integer                   :: nstep_max, nstep_run

  type(carma_type), target            :: carma
  type(carma_type), pointer           :: carma_ptr
  type(carmastate_type)               :: cstate
  integer                             :: rc = 0

  real(kind=f), allocatable   :: zc(:), zl(:), p(:), pl(:), t(:), rho(:)
  real(kind=f), allocatable   :: mmr(:,:,:), mmr_gas(:,:), new_gas(:,:)
  real(kind=f), allocatable   :: satliq(:,:), satice(:,:)
  real(kind=f), allocatable   :: r(:), rmass(:)

  real(kind=f)          :: lat, lon
  integer               :: i, istep, igas, ielem, ibin
  real(kind=f)          :: time
  real(kind=f)          :: rmin, rmrat, RHO_SULFATE
  real(kind=f)          :: log_r, log_mu_cm, log_sigma, norm, r_cm

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: <scenario_file> <output_dir> [nstep_max]'
    call exit(2)
  end if
  call get_command_argument(1, scenario_path)
  call get_command_argument(2, output_dir)
  nstep_max = nstep
  if (command_argument_count() >= 3) then
    call get_command_argument(3, nstep_max_str)
    read(nstep_max_str, *, iostat=ios) nstep_max
    if (ios /= 0 .or. nstep_max < 1) nstep_max = nstep
  end if
  nstep_run = min(nstep, nstep_max)

  open(newunit=unit_in, file=trim(scenario_path), action='read', &
       status='old', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open scenario file: ', trim(scenario_path)
    call exit(3)
  end if
  read(unit_in, *, iostat=ios) T_scen, p_scen_hPa, rh_scen, &
                               h2so4_pptv, mu_nm, sigma_g
  close(unit_in)
  if (ios /= 0) then
    write(0, '(A)') 'malformed scenario file'
    call exit(4)
  end if

  allocate(zc(NZ), zl(NZP1), p(NZ), pl(NZP1), t(NZ), rho(NZ))
  allocate(mmr(NZ,NELEM,NBIN))
  allocate(mmr_gas(NZ,NGAS), new_gas(NZ,NGAS))
  allocate(satliq(NZ,NGAS), satice(NZ,NGAS))
  allocate(r(NBIN), rmass(NBIN))

  call CARMA_Create(carma, NBIN, NELEM, NGROUP, NSOLUTE, NGAS, NWAVE, rc, &
      LUNOPRT=LUNOPRT)
  if (rc /= 0) stop '*** CARMA_Create FAILED ***'
  carma_ptr => carma

  rmrat = 2._f
  rmin  = 2.e-8_f
  RHO_SULFATE = 1.923_f

  call CARMAGROUP_Create(carma, 1, 'sulfate', rmin, rmrat, I_SPHERE, 1._f, &
      .false., rc, irhswell=I_WTPCT_H2SO4, do_drydep=.true., &
      shortname='SULF', is_sulfate=.true.)
  if (rc /= 0) stop '*** CARMAGROUP_Create FAILED ***'

  call CARMAELEMENT_Create(carma, 1, 1, 'Sulfate', RHO_SULFATE, I_VOLATILE, &
      I_H2SO4, rc, shortname='SULF')
  if (rc /= 0) stop '*** CARMAELEMENT_Create FAILED ***'

  call CARMAGAS_Create(carma, 1, 'Water Vapor', WTMOL_H2O, &
      I_VAPRTN_H2O_MURPHY2005, I_GCOMP_H2O, rc, shortname='Q', &
      dgc_threshold=0.1_f, ds_threshold=0.1_f)
  if (rc /= 0) stop '*** CARMAGAS_Create FAILED ***'

  call CARMAGAS_Create(carma, 2, 'Sulpheric Acid', 98.078479_f, &
      I_VAPRTN_H2SO4_AYERS1980, I_GCOMP_H2SO4, rc, shortname='H2SO4', &
      dgc_threshold=0.1_f, ds_threshold=0.1_f)
  if (rc /= 0) stop '*** CARMAGAS_Create FAILED ***'

  call CARMA_AddGrowth(carma, 1, 2, rc)
  if (rc /= 0) stop '*** CARMA_AddGrowth FAILED ***'
  call CARMA_AddNucleation(carma, 1, 1, I_HOMNUC, 0._f, rc, igas=2)
  if (rc /= 0) stop '*** CARMA_AddNucleation FAILED ***'
  call CARMA_AddCoagulation(carma, 1, 1, 1, I_COLLEC_FUCHS, rc)
  if (rc /= 0) stop '*** CARMA_AddCoagulation FAILED ***'

  call CARMA_Initialize(carma, rc, do_grow=.true., do_coag=.true., &
      do_substep=.true., do_thermo=.true., maxretries=16, maxsubsteps=32, &
      dt_threshold=1._f, sulfnucl_method='ZhaoTurco')
  if (rc /= 0) stop '*** CARMA_Initialize FAILED ***'

  lat = -40.0_f
  lon = -105.0_f
  do i = 1, NZ
    zc(i) = zmin + (deltaz * (i - 0.5_f))
  end do
  call GetStandardAtmosphere(zc, p=p, t=t)
  do i = 1, NZP1
    zl(i) = zmin + ((i - 1) * deltaz)
  end do
  call GetStandardAtmosphere(zl, p=pl)

  ! --- Substitute scenario values ---
  t(1)  = T_scen
  p(1)  = p_scen_hPa * 100._f
  zl(1) = zc(1) - deltaz
  zl(2) = zc(1) + deltaz
  rho(1) = (p(1) * 10._f) / (R_AIR * t(1)) * (1e-3_f * 1e6_f)
  pl(1) = p(1) - (zl(1) - zc(1)) * rho(1) * (GRAV / 100._f)
  pl(2) = p(1) - (zl(2) - zc(1)) * rho(1) * (GRAV / 100._f)

  mmr_gas(:,1) = rh_scen * exp(54.842763_f - 6763.22_f/t(1) &
                               - 4.210_f*log(t(1)) + 0.000367_f*t(1)) &
                 * 18._f / (29._f * p(1))
  mmr_gas(:,2) = h2so4_pptv * 1.e-12_f * (98._f / 29._f)

  satliq(:,:)   = -1._f
  satice(:,:)   = -1._f
  mmr(:,:,:)    = 0._f

  call CARMAGROUP_Get(carma, 1, rc, r=r, rmass=rmass)
  log_mu_cm = log(mu_nm * 1.e-7_f)
  log_sigma = log(sigma_g)
  norm = 0._f
  do ibin = 1, NBIN
    r_cm = r(ibin)
    log_r = log(r_cm)
    mmr(1,1,ibin) = exp(-0.5_f * ((log_r - log_mu_cm) / log_sigma)**2) &
                    / (r_cm * log_sigma * sqrt(2._f * PI)) &
                    * rmass(ibin)
    norm = norm + mmr(1,1,ibin)
  end do
  if (norm > 0._f) then
    mmr(1,1,:) = mmr(1,1,:) * (1.e-18_f / norm)
  end if

  do istep = 1, nstep_run
    time = (istep - 1) * dtime
    call CARMASTATE_Create(cstate, carma_ptr, time, dtime, NZ, &
        I_CART, lat, lon, zc(:), zl(:), p(:), pl(:), t(:), rc, &
        told=t(:), qh2o=mmr_gas(1,:))
    if (rc /= 0) stop '*** CARMASTATE_Create FAILED ***'
    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_SetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do
    new_gas = mmr_gas(:,:)
    do igas = 1, NGAS
      call CARMASTATE_SetGas(cstate, igas, new_gas(:,igas), rc, &
          mmr_old=mmr_gas(:,igas), &
          satice_old=satice(:,igas), satliq_old=satliq(:,igas))
    end do

    call CARMASTATE_Step(cstate, rc)
    if (rc /= 0) stop '*** CARMASTATE_Step FAILED ***'

    ! ---- DUMP per-substep state (internal procedure, see CONTAINS) ----
    call dump_substep(istep, cstate, output_dir)

    call CARMASTATE_GetState(cstate, rc, t=t(:))
    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_GetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do
    do igas = 1, NGAS
      call CARMASTATE_GetGas(cstate, igas, mmr_gas(:,igas), rc, &
          satliq=satliq(:,igas), satice=satice(:,igas))
    end do
  end do

  call CARMASTATE_Destroy(cstate, rc)
  call CARMA_Destroy(carma, rc)
  deallocate(zc, zl, p, pl, t, rho, mmr, mmr_gas, new_gas)
  deallocate(satliq, satice, r, rmass)

contains

  !! Dump every microfast/microslow intermediate array to per-substep
  !! binary stream files. Layout: <output_dir>/substep_<NNNN>_<name>.bin.
  subroutine dump_substep(istep_d, cs, outdir)
    integer, intent(in)                  :: istep_d
    type(carmastate_type), intent(inout) :: cs
    character(len=*), intent(in)         :: outdir
    character(len=512)                   :: prefix

    write(prefix, '(A,A,I4.4,A)') trim(outdir), '/substep_', istep_d, '_'

    ! Primary state
    call dump_3d(prefix, 'pc', cs%f_pc)
    call dump_2d(prefix, 'gc', cs%f_gc)
    call dump_1d(prefix, 't', cs%f_t)
    call dump_1d(prefix, 'p', cs%f_p)
    call dump_1d(prefix, 'rhoa', cs%f_rhoa)
    call dump_1d(prefix, 'zmet', cs%f_zmet)
    call dump_1d(prefix, 'wtpct', cs%f_wtpct)
    ! Probes: scalar sulfate_density / sulfate_surf_tens at the
    ! (wtpct, t) of this substep. JAX side feeds the same (wtpct, t)
    ! and compares against these.
    call dump_sulfate_probes(prefix, cs)

    ! Vapor / saturation
    call dump_2d(prefix, 'pvapl', cs%f_pvapl)
    call dump_2d(prefix, 'pvapi', cs%f_pvapi)
    call dump_2d(prefix, 'supsatl', cs%f_supsatl)
    call dump_2d(prefix, 'supsati', cs%f_supsati)
    call dump_2d(prefix, 'akelvin', cs%f_akelvin)
    call dump_2d(prefix, 'akelvini', cs%f_akelvini)

    ! Growth kernel
    call dump_3d(prefix, 'gro', cs%f_gro)
    call dump_3d(prefix, 'gro1', cs%f_gro1)

    ! Microfast intermediate rates
    call dump_2d(prefix, 'rhompe', cs%f_rhompe)
    call dump_3d(prefix, 'rnuclg', cs%f_rnuclg)
    call dump_2d(prefix, 'rnucpe', cs%f_rnucpe)
    call dump_2d(prefix, 'growpe', cs%f_growpe)
    call dump_2d(prefix, 'evappe', cs%f_evappe)
    call dump_2d(prefix, 'growlg', cs%f_growlg)
    call dump_2d(prefix, 'evaplg', cs%f_evaplg)

    ! Gas / heat production
    call dump_1d(prefix, 'gasprod', cs%f_gasprod)
    call dump_1d(prefix, 'rlheat', cs%f_rlheat)

    ! Coag (microslow) intermediates
    call dump_2d(prefix, 'pconmax', cs%f_pconmax)
    call dump_3d(prefix, 'coaglg', cs%f_coaglg)
    call dump_3d(prefix, 'coagpe', cs%f_coagpe)

    ! Prestep snapshots (state at start of substep, used by bookkeeping)
    call dump_3d(prefix, 'pcl', cs%f_pcl)
    call dump_2d(prefix, 'gcl', cs%f_gcl)
    call dump_1d(prefix, 'told', cs%f_told)
    call dump_2d(prefix, 'd_gc', cs%f_d_gc)
    call dump_1d(prefix, 'd_t', cs%f_d_t)

    ! Aerosol kinematics (for setup_ckern bench)
    call dump_1d(prefix, 'rmu', cs%f_rmu)
    call dump_3d(prefix, 'bpm', cs%f_bpm)
    call dump_3d(prefix, 're',  cs%f_re)
    call dump_3d(prefix, 'vf',  cs%f_vf)

    ! Wet radius / density (rhopart + getwetr outputs)
    call dump_1d(prefix, 'relhum', cs%f_relhum)
    call dump_3d(prefix, 'rhop', cs%f_rhop)
    call dump_3d(prefix, 'r_wet', cs%f_r_wet)
    call dump_3d(prefix, 'rhop_wet', cs%f_rhop_wet)
    call dump_3d(prefix, 'rup_wet', cs%f_rup_wet)

    ! Auxiliary state used by gasexchange (cmf, totevap as float for binary I/O).
    call dump_2d(prefix, 'cmf', cs%f_cmf)
    call dump_totevap(prefix, cs)

    ! Per-substep probe: invoke binary_nuc_zhao1995 directly with the
    ! end-of-step cstate so the JAX bench has matching inputs/outputs
    ! that bypass the multi-substep gsolve/tsolve evolution complication.
    call dump_zhao1995_probe(prefix, cs)

    ! Per-substep probe: growp at end-of-step state (matched to end-of-
    ! step pc + growlg from growevapl probe). upgxfer is a no-op in
    ! sulfate scope (rnuclg=0 throughout).
    call dump_growp_probe(prefix, cs)

    ! Per-substep probe: totalcondensate + gsolve at end-of-step state.
    ! Internally saves/restores pc, gc, t so other probes remain valid.
    call dump_gsolve_probe(prefix, cs)

    ! Per-substep probe: gsolve + tsolve at end-of-step state. Dumps
    ! rlhe/rlhm (latent-heat coefs), rlprod (after gsolve), t_pre, t_post.
    ! Internally saves/restores cstate.
    call dump_tsolve_probe(prefix, cs)

    ! Per-substep probe: nsubsteps at end-of-step state. Dumps the
    ! suggested ntsubsteps integer (cast to f8).
    call dump_nsubsteps_probe(prefix, cs)

    ! Per-substep probe: full microslow chain (coagl → per-bin
    ! coagp/csolve) at end-of-step state. Dumps coaglg/coagpe/pc
    ! before & after, lets JAX bench each piece.
    call dump_microslow_probe(prefix, cs)

    ! Per-substep probe: full microfast at end-of-step state. Mirrors
    ! microfast.F90: zeromicro → totalcondensate → supersat → sulfnuc →
    ! growevapl → cloud nuc/freeze (no-op for sulfate) → growp/upgxfer/
    ! psolve loop → evapp → downgxfer → downgevapply → gsolve → tsolve.
    ! Saves/restores cstate. Dumps pc/gc/t before & after.
    call dump_microfast_probe(prefix, cs)

    ! Per-substep probe: full microfast prefix at end-of-step state up
    ! through psolve. Mirrors microfast.F90: sulfnuc → growevapl →
    ! per-bin growp → psolve. Dumps pc_postpsolve_probe (NBIN, NELEM at iz).
    call dump_psolve_probe(prefix, cs)

    ! Per-substep probe: evapp + downgevapply at end-of-step state.
    ! Mirrors microfast's growevapl→growp→upgxfer→psolve→evapp→downg
    ! sequence. JAX bench feeds end-of-step pc+evaplg and verifies
    ! evappe (from evapp) + pc_after_apply (from downgevapply).
    call dump_evapp_probe(prefix, cs)

    ! Per-substep probe: invoke maxconc then growevapl at end-of-step
    ! state and dump growlg + evaplg (NBIN,NGROUP). The dumped growlg/
    ! evaplg are from microfast and reflect a pre-advance pc state;
    ! this probe gives a properly matched (pc, growlg, evaplg) triple.
    call dump_growevapl_probe(prefix, cs)

    ! Per-substep probe: invoke pheat per bin and dump dmdt (NBIN,).
    ! pheat computes the mass growth rate for one particle in each bin,
    ! including optional particle heating (do_pheat=false in sulfate test,
    ! so the simple Köhler formula runs). dmdt is the key output that feeds
    ! growevapl but is not stored in cstate directly.
    call dump_pheat_probe(prefix, cs)

    ! Per-substep probe: invoke maxconc at end-of-step state and dump the
    ! result. pconmax is computed pre-microfast inside newstate_calc.F90:175
    ! so the existing cstate%f_pconmax doesn't correspond to the end-of-step
    ! pc dump. This probe gives a correctly matched (pc, pconmax) pair.
    call dump_maxconc_probe(prefix, cs)

    ! Per-substep probe: invoke Fortran's gasexchange directly with the
    ! end-of-step cstate. gasexchange is commented out in modern microfast
    ! (gsolve uses total-condensate instead) — this probe is purely for
    ! kernel-level validation of the JAX gasexchange port.
    call dump_gasexchange_probe(prefix, cs)

    ! Static carma PPM and growth tables. Dumped at step 1 only.
    ! only — these don't change across substeps.
    if (istep_d == 1) then
      call dump_carma_bins(prefix)
      call dump_carma_ppm(prefix)
      call dump_5d(prefix, 'ckernel', cs%f_ckernel)
    end if
  end subroutine dump_substep


  !! Probe: call binary_nuc_zhao1995 with end-of-step cstate inputs
  !! and dump the four scalar outputs (nucrate_cgs, mass_cluster_dry,
  !! radius_cluster, ftry). JAX bench then feeds the same inputs and
  !! compares output bit-for-bit. Sulfate test is single-cell (NZ=1)
  !! so we probe iz=1 only.
  subroutine dump_zhao1995_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f) :: probe(4)            ! [nucrate_cgs, mass_cluster_dry, rstar, ftry]
    real(kind=f) :: temp, rh, beta1, beta2, rb
    real(kind=f) :: h2o_cgs, h2so4_cgs, h2o_n, h2so4_n
    real(kind=f) :: nucrate_cgs, mass_cluster_dry, radius_cluster, ftry
    integer      :: rc_loc, igash2o, igash2so4
    integer, parameter :: iz = 1
    interface
      subroutine binary_nuc_zhao1995(carma, cstate, temp, weight_percent, rh, &
                                     h2so4, h2so4_cgs, h2o, h2o_cgs, beta1, &
                                     nucrate_cgs, mass_cluster_dry, radius_cluster, &
                                     ftry, rc)
        use carma_precision_mod
        use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        real(kind=f), intent(in)             :: temp, weight_percent, rh
        real(kind=f), intent(in)             :: h2so4, h2so4_cgs, h2o, h2o_cgs, beta1
        real(kind=f), intent(out)            :: nucrate_cgs, radius_cluster, mass_cluster_dry, ftry
        integer, intent(inout)               :: rc
      end subroutine binary_nuc_zhao1995
    end interface

    if (.not. allocated(cs%f_t) .or. .not. allocated(cs%f_gc) .or. &
        .not. allocated(cs%f_supsatl) .or. .not. allocated(cs%f_wtpct)) return

    ! Gas indices: 1=H2O, 2=H2SO4 in carma_sulfatetest.F90
    igash2o   = 1
    igash2so4 = 2

    temp = cs%f_t(iz)
    rb   = RGAS * temp / 2._f / PI
    beta1 = sqrt(rb / carma%f_gas(igash2so4)%f_wtmol)
    beta2 = sqrt(rb / carma%f_gas(igash2o)%f_wtmol)

    h2so4_cgs = cs%f_gc(iz, igash2so4) / cs%f_zmet(iz)
    h2o_cgs   = cs%f_gc(iz, igash2o)   / cs%f_zmet(iz)
    h2so4_n   = h2so4_cgs * AVG / carma%f_gas(igash2so4)%f_wtmol
    h2o_n     = h2o_cgs   * AVG / carma%f_gas(igash2o)%f_wtmol
    rh        = cs%f_supsatl(iz, igash2o) + 1._f

    nucrate_cgs = 0._f
    mass_cluster_dry = 0._f
    radius_cluster = 0._f
    ftry = 0._f
    rc_loc = 0
    call binary_nuc_zhao1995(carma, cs, temp, cs%f_wtpct(iz), rh, &
                              h2so4_n, h2so4_cgs, h2o_n, h2o_cgs, beta1, &
                              nucrate_cgs, mass_cluster_dry, radius_cluster, &
                              ftry, rc_loc)

    probe(1) = nucrate_cgs
    probe(2) = mass_cluster_dry
    probe(3) = radius_cluster
    probe(4) = ftry
    call dump_alloc_1d(prefix, 'zhao1995_probe', probe)
  end subroutine dump_zhao1995_probe


  !! Probe: invoke Fortran's gasexchange directly with end-of-step cstate
  !! and dump the resulting cstate%f_gasprod (NGAS,). gasexchange is
  !! commented out in modern microfast.F90:152 (gsolve overwrites
  !! gasprod via total-condensate), so this probe is the only way to
  !! bench the JAX gasexchange port against Fortran. We zero gasprod
  !! before the call so the output reflects ONLY gasexchange's
  !! contribution (gasexchange uses += internally).
  !!
  !! Side effect: cstate%f_gasprod is overwritten with gasexchange's
  !! output. Since this is end-of-step and the test driver doesn't read
  !! gasprod afterwards, this is harmless.
  !! Probe: call maxconc with end-of-step pc/zmet and dump the result.
  !! pconmax in the regular dump reflects the pre-microfast call inside
  !! newstate_calc.F90:175, which is computed on pre-microfast pc. This
  !! probe gives end-of-step (pc, zmet) → pconmax so the JAX bench
  !! has a matched pair.
  !! Probe: call pheat for each bin at end-of-step state and dump the
  !! resulting dmdt array (NBIN,). Sulfate test: NGROUP=1, igas=igash2so4,
  !! NZ=1. do_pheat=false and NWAVE=0, so the simple no-radiation branch
  !! runs: dmdt = pvap*(ss+1-akas)*gro/(1+gro*gro1*pvap).
  !! Probe: call maxconc then growevapl at end-of-step cstate and dump
  !! the resulting growlg and evaplg arrays (NBIN, NGROUP). maxconc is
  !! called first so pconmax reflects the end-of-step pc (otherwise
  !! growevapl's FEW_PC gate might use stale pconmax from pre-microfast).
  !! Dump the PPM growth tables from the carma object (static, step-1 only).
  !! Shapes: dm (NBIN,NGROUP), pratt (3,NBIN,NGROUP), prat (4,NBIN,NGROUP),
  !!         pden1 (NBIN,NGROUP), palr (4,NGROUP), igrowgas (NELEM).
  !! Probe: compute growlg via maxconc+growevapl then run growp for all
  !! (ibin, ielem) and dump the resulting growpe (NBIN, NELEM).
  !! Must rebuild growlg first because growlg gets zeroed between steps.
  subroutine dump_growp_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    integer                                 :: rc_loc, ib, ie
    integer, parameter                      :: iz = 1
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz; integer, intent(inout) :: rc
      end subroutine maxconc
      subroutine growevapl(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz; integer, intent(inout) :: rc
      end subroutine growevapl
      subroutine growp(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz, ibin, ielem
        integer, intent(inout)               :: rc
      end subroutine growp
    end interface

    if (.not. allocated(cs%f_growpe)) return
    cs%f_growlg(:,:) = 0._f; cs%f_evaplg(:,:) = 0._f; cs%f_growpe(:,:) = 0._f
    rc_loc = 0
    call maxconc(carma, cs, iz, rc_loc)
    if (rc_loc < 0) rc_loc = 0
    call growevapl(carma, cs, iz, rc_loc)
    if (rc_loc < 0) rc_loc = 0
    do ie = 1, carma%f_NELEM
      do ib = 1, carma%f_NBIN
        call growp(carma, cs, iz, ib, ie, rc_loc)
        if (rc_loc < 0) rc_loc = 0
      end do
    end do
    call dump_alloc_2d(prefix, 'growpe_probe', cs%f_growpe)
  end subroutine dump_growp_probe


  !! Probe: totalcondensate + gsolve at end-of-step state.
  !! 1. Compute previous_ice/previous_liquid via totalcondensate on the
  !!    starting cstate (the dump's pc).
  !! 2. Run the full microfast evolution sequence (sulfnuc → growevapl →
  !!    psolve loop → evapp → downgevapply) to produce the post-evolution pc.
  !! 3. Compute total_ice/total_liquid via totalcondensate on post pc.
  !! 4. Snapshot gc[iz,:], call gsolve, dump gc_pregsolve, gc_postgsolve.
  !! 5. Also dumps previous_ice, total_ice, previous_liquid, total_liquid.
  subroutine dump_gsolve_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: prev_ice(:), prev_liq(:)
    real(kind=f), allocatable               :: tot_ice(:), tot_liq(:)
    real(kind=f), allocatable               :: gc_pre(:), gc_post(:)
    integer                                 :: rc_loc, ie, ib, ng
    integer, parameter                      :: iz = 1
    real(kind=f), parameter                 :: scale_thr = 1._f
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine maxconc
      subroutine sulfnuc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine sulfnuc
      subroutine growevapl(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine growevapl
      subroutine growp(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine growp
      subroutine upgxfer(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine upgxfer
      subroutine psolve(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine psolve
      subroutine evapp(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine evapp
      subroutine downgxfer(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine downgxfer
      subroutine downgevapply(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine downgevapply
      subroutine totalcondensate(carma, cstate, iz, total_ice, total_liquid, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f), intent(out) :: total_ice(2), total_liquid(2)  ! NGAS=2 in sulfate test
        integer, intent(inout) :: rc
      end subroutine totalcondensate
      subroutine gsolve(carma, cstate, iz, previous_ice, previous_liquid, scale_threshold, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f), intent(in) :: previous_ice(2), previous_liquid(2)  ! NGAS=2
        real(kind=f) :: scale_threshold
        integer, intent(inout) :: rc
      end subroutine gsolve
    end interface

    if (.not. allocated(cs%f_gc)) return
    ng = carma%f_NGAS
    allocate(prev_ice(ng), prev_liq(ng), tot_ice(ng), tot_liq(ng))
    allocate(gc_pre(ng), gc_post(ng))

    ! Save state that this probe will mutate, restore at end
    block
      real(kind=f), allocatable :: pc_save(:,:,:), gc_save(:,:), t_save(:)
      real(kind=f), allocatable :: rlheat_save(:)
      allocate(pc_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      allocate(gc_save(size(cs%f_t), ng), t_save(size(cs%f_t)))
      allocate(rlheat_save(size(cs%f_t)))
      pc_save = cs%f_pc; gc_save = cs%f_gc; t_save = cs%f_t
      rlheat_save = cs%f_rlheat

    ! Step 1: previous_ice/liquid from starting pc
    rc_loc = 0
    call totalcondensate(carma, cs, iz, prev_ice, prev_liq, rc_loc); if (rc_loc<0) rc_loc=0

    ! Step 2: full microfast evolution
    cs%f_growlg(:,:) = 0._f; cs%f_evaplg(:,:) = 0._f
    cs%f_growpe(:,:) = 0._f; cs%f_evappe(:,:) = 0._f
    cs%f_rnucpe(:,:) = 0._f; cs%f_rhompe(:,:) = 0._f
    cs%f_rnuclg(:,:,:) = 0._f
    call maxconc(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
    call sulfnuc(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
    call growevapl(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
    do ie = 1, carma%f_NELEM
      do ib = 1, carma%f_NBIN
        call growp(carma, cs, iz, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
        call upgxfer(carma, cs, iz, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
        call psolve(carma, cs, iz, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
      end do
    end do
    call evapp(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
    call downgxfer(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
    call downgevapply(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0

    ! Step 3: total_ice/liquid from post pc
    call totalcondensate(carma, cs, iz, tot_ice, tot_liq, rc_loc); if (rc_loc<0) rc_loc=0

    ! Step 4: snapshot gc, run gsolve, snapshot gc again
    gc_pre = cs%f_gc(iz, :)
    call gsolve(carma, cs, iz, prev_ice, prev_liq, scale_thr, rc_loc); if (rc_loc<0) rc_loc=0
    gc_post = cs%f_gc(iz, :)

    call dump_alloc_1d(prefix, 'previous_ice_probe',    prev_ice)
    call dump_alloc_1d(prefix, 'previous_liquid_probe', prev_liq)
    call dump_alloc_1d(prefix, 'total_ice_probe',       tot_ice)
    call dump_alloc_1d(prefix, 'total_liquid_probe',    tot_liq)
    call dump_alloc_1d(prefix, 'gc_pregsolve_probe',    gc_pre)
    call dump_alloc_1d(prefix, 'gc_postgsolve_probe',   gc_post)

    ! Restore mutated state so subsequent probes start from dump's cstate.
    cs%f_pc = pc_save; cs%f_gc = gc_save; cs%f_t = t_save
    cs%f_rlheat = rlheat_save
    deallocate(pc_save, gc_save, t_save, rlheat_save)
    end block

    deallocate(prev_ice, prev_liq, tot_ice, tot_liq, gc_pre, gc_post)
  end subroutine dump_gsolve_probe


  !! Probe: invoke Fortran's microfast directly at end-of-step state and
  !! dump pc / gc / t before and after. Saves/restores cstate. dtime is
  !! the test's substep size = dtime_orig / 1 (single substep at end-of-step).
  subroutine dump_microfast_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: pc_pre(:,:), pc_post(:,:)
    real(kind=f), allocatable               :: gc_pre(:), gc_post(:)
    real(kind=f) :: t_pre, t_post
    integer                                 :: rc_loc, ie, ib
    integer, parameter                      :: iz = 1
    real(kind=f), parameter                 :: scale_thr = 1._f
    interface
      subroutine microfast(carma, cstate, iz, scale_threshold, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f) :: scale_threshold
        integer, intent(inout) :: rc
      end subroutine microfast
    end interface

    if (.not. allocated(cs%f_pc)) return
    allocate(pc_pre(carma%f_NBIN, carma%f_NELEM))
    allocate(pc_post(carma%f_NBIN, carma%f_NELEM))
    allocate(gc_pre(carma%f_NGAS), gc_post(carma%f_NGAS))

    block
      real(kind=f), allocatable :: pc_save(:,:,:), gc_save(:,:), t_save(:)
      real(kind=f), allocatable :: rlheat_save(:)
      real(kind=f) :: rlprod_save
      allocate(pc_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      allocate(gc_save(size(cs%f_t), carma%f_NGAS), t_save(size(cs%f_t)))
      allocate(rlheat_save(size(cs%f_t)))
      pc_save = cs%f_pc; gc_save = cs%f_gc; t_save = cs%f_t
      rlheat_save = cs%f_rlheat; rlprod_save = cs%f_rlprod

      ! Snapshot pc/gc/t before microfast
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_pre(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      gc_pre = cs%f_gc(iz, :)
      t_pre = cs%f_t(iz)

      ! Run microfast (which internally zeromicros & does the full chain)
      rc_loc = 0
      call microfast(carma, cs, iz, scale_thr, rc_loc); if (rc_loc<0) rc_loc=0

      ! Snapshot pc/gc/t after microfast
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_post(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      gc_post = cs%f_gc(iz, :)
      t_post = cs%f_t(iz)

      ! Restore
      cs%f_pc = pc_save; cs%f_gc = gc_save; cs%f_t = t_save
      cs%f_rlheat = rlheat_save; cs%f_rlprod = rlprod_save
      deallocate(pc_save, gc_save, t_save, rlheat_save)
    end block

    call dump_alloc_2d(prefix, 'pc_premicrofast_probe',  pc_pre)
    call dump_alloc_2d(prefix, 'pc_postmicrofast_probe', pc_post)
    call dump_alloc_1d(prefix, 'gc_premicrofast_probe',  gc_pre)
    call dump_alloc_1d(prefix, 'gc_postmicrofast_probe', gc_post)
    block
      real(kind=f) :: tarr(2)
      tarr(1) = t_pre; tarr(2) = t_post
      call dump_alloc_1d(prefix, 't_microfast_probe', tarr)
    end block

    deallocate(pc_pre, pc_post, gc_pre, gc_post)
  end subroutine dump_microfast_probe


  !! Probe: full microslow chain at end-of-step state. Saves/restores
  !! cstate. Sequence: zero coag accumulators, call coagl, then per
  !! (ielem, ibin) call coagp + csolve. Dumps:
  !!   pc_premicroslow_probe (NBIN, NELEM at iz)
  !!   coaglg_probe (NBIN, NGROUP) — output of coagl alone
  !!   coagpe_probe (NBIN, NELEM)  — accumulated by coagp before csolve
  !!   pc_postmicroslow_probe (NBIN, NELEM at iz)
  subroutine dump_microslow_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: pc_snap(:,:)
    real(kind=f), allocatable               :: coaglg_iz(:,:), coagpe_iz(:,:)
    integer                                 :: rc_loc, ie, ib, i
    integer, parameter                      :: iz = 1
    interface
      subroutine coagl(carma, cstate, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(inout) :: rc
      end subroutine coagl
      subroutine coagp(carma, cstate, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: ibin, ielem; integer, intent(inout) :: rc
      end subroutine coagp
      subroutine csolve(carma, cstate, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: ibin, ielem; integer, intent(inout) :: rc
      end subroutine csolve
    end interface

    if (.not. allocated(cs%f_pc) .or. .not. allocated(cs%f_coaglg)) return
    block
      real(kind=f), allocatable :: pc_save(:,:,:)
      real(kind=f), allocatable :: coaglg_save(:,:,:), coagpe_save(:,:,:)
      allocate(pc_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      allocate(coaglg_save(size(cs%f_t), carma%f_NBIN, carma%f_NGROUP))
      allocate(coagpe_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      pc_save = cs%f_pc
      coaglg_save = cs%f_coaglg
      coagpe_save = cs%f_coagpe

      ! Snapshot pc before microslow
      allocate(pc_snap(carma%f_NBIN, carma%f_NELEM))
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_snap(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      call dump_alloc_2d(prefix, 'pc_premicroslow_probe', pc_snap)

      ! Mirror microslow: zero accumulators, call coagl, then loop coagp+csolve
      cs%f_coagpe(:,:,:) = 0._f
      cs%f_coaglg(:,:,:) = 0._f
      rc_loc = 0
      call coagl(carma, cs, rc_loc); if (rc_loc<0) rc_loc=0

      ! Dump coaglg from this iz (before any csolve mutates pc)
      allocate(coaglg_iz(carma%f_NBIN, carma%f_NGROUP))
      do i = 1, carma%f_NGROUP; do ib = 1, carma%f_NBIN
        coaglg_iz(ib, i) = cs%f_coaglg(iz, ib, i)
      end do; end do
      call dump_alloc_2d(prefix, 'coaglg_probe', coaglg_iz)
      deallocate(coaglg_iz)

      ! Run coagp + csolve per (ielem, ibin), then dump coagpe at this iz
      ! Note: coagp at (ibin, ielem) accumulates ONLY contributions to that
      ! cell, but writes to all iz. For our NZ=1 case, it's just (ibin, ielem).
      ! We capture coagpe AFTER all coagp calls finish, since csolve doesn't
      ! reset it.
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          call coagp(carma, cs, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
          call csolve(carma, cs, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
        end do
      end do

      ! Dump coagpe at this iz
      allocate(coagpe_iz(carma%f_NBIN, carma%f_NELEM))
      do ie = 1, carma%f_NELEM; do ib = 1, carma%f_NBIN
        coagpe_iz(ib, ie) = cs%f_coagpe(iz, ib, ie)
      end do; end do
      call dump_alloc_2d(prefix, 'coagpe_probe', coagpe_iz)
      deallocate(coagpe_iz)

      ! Snapshot pc after csolve
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_snap(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      call dump_alloc_2d(prefix, 'pc_postmicroslow_probe', pc_snap)
      deallocate(pc_snap)

      ! Restore
      cs%f_pc = pc_save
      cs%f_coaglg = coaglg_save
      cs%f_coagpe = coagpe_save
      deallocate(pc_save, coaglg_save, coagpe_save)
    end block
  end subroutine dump_microslow_probe


  !! Probe: nsubsteps at end-of-step state. Saves/restores cstate.
  !! Calls maxconc first so pconmax matches end-of-step pc, then nsubsteps.
  !! Dumps:
  !!   nsubsteps_probe : 1-element array with the suggested ntsubsteps as f8.
  subroutine dump_nsubsteps_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    integer                                 :: rc_loc, nts
    real(kind=f)                            :: dtime_local
    integer, parameter                      :: iz = 1
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine maxconc
      subroutine nsubsteps(carma, cstate, iz, dtime_save, ntsubsteps, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f), intent(in) :: dtime_save
        integer, intent(inout) :: ntsubsteps, rc
      end subroutine nsubsteps
    end interface

    if (.not. allocated(cs%f_pc)) return

    block
      real(kind=f), allocatable :: pconmax_save(:,:)
      allocate(pconmax_save(size(cs%f_t), carma%f_NGROUP))
      pconmax_save = cs%f_pconmax

      rc_loc = 0
      call maxconc(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0

      ! Use the test's outer dtime (1800s)
      dtime_local = cs%f_dtime_orig
      nts = 1
      call nsubsteps(carma, cs, iz, dtime_local, nts, rc_loc); if (rc_loc<0) rc_loc=0

      cs%f_pconmax = pconmax_save
      deallocate(pconmax_save)
    end block

    block
      real(kind=f) :: arr(1)
      arr(1) = real(nts, kind=f)
      call dump_alloc_1d(prefix, 'nsubsteps_probe', arr)
    end block
  end subroutine dump_nsubsteps_probe


  !! Probe: full evolution → gsolve → tsolve at end-of-step state.
  !! Dumps:
  !!   t_pretsolve_probe  : t[iz] before tsolve (after gsolve sets rlprod)
  !!   t_posttsolve_probe : t[iz] after tsolve
  !!   rlprod_probe       : rlprod scalar (set by gsolve)
  !!   rlheat_pre_probe   : rlheat[iz] before tsolve
  !!   rlheat_post_probe  : rlheat[iz] after tsolve
  !!   rlhe_probe         : rlhe[iz, :] (NGAS,)
  !!   rlhm_probe         : rlhm[iz, :] (NGAS,)
  !! Saves/restores cstate.
  subroutine dump_tsolve_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: prev_ice(:), prev_liq(:)
    real(kind=f), allocatable               :: scal_a(:), scal_b(:)
    real(kind=f), allocatable               :: rlhe1(:), rlhm1(:)
    real(kind=f) :: t_pre, t_post, rlheat_pre, rlheat_post, rlprod_val
    integer                                 :: rc_loc, ie, ib, ng
    integer, parameter                      :: iz = 1
    real(kind=f), parameter                 :: scale_thr = 1._f
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine maxconc
      subroutine sulfnuc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine sulfnuc
      subroutine growevapl(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine growevapl
      subroutine growp(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine growp
      subroutine upgxfer(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine upgxfer
      subroutine psolve(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine psolve
      subroutine evapp(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine evapp
      subroutine downgxfer(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine downgxfer
      subroutine downgevapply(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine downgevapply
      subroutine totalcondensate(carma, cstate, iz, total_ice, total_liquid, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f), intent(out) :: total_ice(2), total_liquid(2)
        integer, intent(inout) :: rc
      end subroutine totalcondensate
      subroutine gsolve(carma, cstate, iz, previous_ice, previous_liquid, scale_threshold, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f), intent(in) :: previous_ice(2), previous_liquid(2)
        real(kind=f) :: scale_threshold
        integer, intent(inout) :: rc
      end subroutine gsolve
      subroutine tsolve(carma, cstate, iz, scale_threshold, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz
        real(kind=f) :: scale_threshold
        integer, intent(inout) :: rc
      end subroutine tsolve
    end interface

    if (.not. allocated(cs%f_t)) return
    ng = carma%f_NGAS
    allocate(prev_ice(ng), prev_liq(ng), scal_a(ng), scal_b(ng))
    allocate(rlhe1(ng), rlhm1(ng))

    block
      real(kind=f), allocatable :: pc_save(:,:,:), gc_save(:,:), t_save(:)
      real(kind=f), allocatable :: rlheat_save(:)
      real(kind=f) :: rlprod_save
      allocate(pc_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      allocate(gc_save(size(cs%f_t), ng), t_save(size(cs%f_t)))
      allocate(rlheat_save(size(cs%f_t)))
      pc_save = cs%f_pc; gc_save = cs%f_gc; t_save = cs%f_t
      rlheat_save = cs%f_rlheat; rlprod_save = cs%f_rlprod

      ! Run full microfast evolution
      cs%f_growlg(:,:) = 0._f; cs%f_evaplg(:,:) = 0._f
      cs%f_growpe(:,:) = 0._f; cs%f_evappe(:,:) = 0._f
      cs%f_rnucpe(:,:) = 0._f; cs%f_rhompe(:,:) = 0._f
      cs%f_rnuclg(:,:,:) = 0._f
      rc_loc = 0
      call totalcondensate(carma, cs, iz, prev_ice, prev_liq, rc_loc); if (rc_loc<0) rc_loc=0
      call maxconc(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
      call sulfnuc(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
      call growevapl(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          call growp(carma, cs, iz, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
          call upgxfer(carma, cs, iz, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
          call psolve(carma, cs, iz, ib, ie, rc_loc); if (rc_loc<0) rc_loc=0
        end do
      end do
      call evapp(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
      call downgxfer(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
      call downgevapply(carma, cs, iz, rc_loc); if (rc_loc<0) rc_loc=0
      call gsolve(carma, cs, iz, prev_ice, prev_liq, scale_thr, rc_loc); if (rc_loc<0) rc_loc=0

      ! Snapshot t before tsolve
      t_pre = cs%f_t(iz)
      rlheat_pre = cs%f_rlheat(iz)
      rlprod_val = cs%f_rlprod
      ! Get latent-heat coeffs (NGAS at iz)
      rlhe1 = cs%f_rlhe(iz, :)
      rlhm1 = cs%f_rlhm(iz, :)

      call tsolve(carma, cs, iz, scale_thr, rc_loc); if (rc_loc<0) rc_loc=0

      t_post = cs%f_t(iz)
      rlheat_post = cs%f_rlheat(iz)

      ! Restore cstate
      cs%f_pc = pc_save; cs%f_gc = gc_save; cs%f_t = t_save
      cs%f_rlheat = rlheat_save; cs%f_rlprod = rlprod_save
      deallocate(pc_save, gc_save, t_save, rlheat_save)
    end block

    scal_a = (/ t_pre, t_post /)
    scal_b = (/ rlheat_pre, rlheat_post /)
    call dump_alloc_1d(prefix, 't_tsolve_probe',     scal_a)
    call dump_alloc_1d(prefix, 'rlheat_tsolve_probe', scal_b)
    call dump_alloc_1d(prefix, 'rlhe_probe',         rlhe1)
    call dump_alloc_1d(prefix, 'rlhm_probe',         rlhm1)
    block
      real(kind=f) :: rlp(1)
      rlp(1) = rlprod_val
      call dump_alloc_1d(prefix, 'rlprod_probe', rlp)
    end block

    deallocate(prev_ice, prev_liq, scal_a, scal_b, rlhe1, rlhm1)
  end subroutine dump_tsolve_probe


  !! Probe: full microfast prefix (sulfnuc → growevapl → growp/upgxfer →
  !! psolve per (ibin, ielem)) at end-of-step state. Dumps:
  !!   pc_prepsolve_probe  : pc[iz, :, :] before psolve loop (after sulfnuc/growevapl)
  !!   rhompe_probe        : rhompe (NBIN, NELEM) from sulfnuc
  !!   pc_postpsolve_probe : pc[iz, :, :] after psolve loop
  !! JAX bench feeds matching state and runs psolve to verify pc_postpsolve.
  subroutine dump_psolve_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: pc_snap(:,:)
    integer                                 :: rc_loc, ie, ib
    integer, parameter                      :: iz = 1
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine maxconc
      subroutine sulfnuc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine sulfnuc
      subroutine growevapl(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz; integer, intent(inout) :: rc
      end subroutine growevapl
      subroutine growp(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine growp
      subroutine upgxfer(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine upgxfer
      subroutine psolve(carma, cstate, iz, ibin, ielem, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in) :: carma; type(carmastate_type), intent(inout) :: cstate
        integer, intent(in) :: iz, ibin, ielem; integer, intent(inout) :: rc
      end subroutine psolve
    end interface

    if (.not. allocated(cs%f_pc)) return
    block
      real(kind=f), allocatable :: pc_save(:,:,:), gc_save(:,:), t_save(:)
      allocate(pc_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      allocate(gc_save(size(cs%f_t), carma%f_NGAS), t_save(size(cs%f_t)))
      pc_save = cs%f_pc; gc_save = cs%f_gc; t_save = cs%f_t

      ! Zero accumulators that microfast normally zeros at start of substep
      cs%f_growlg(:,:) = 0._f; cs%f_evaplg(:,:) = 0._f
      cs%f_growpe(:,:) = 0._f; cs%f_evappe(:,:) = 0._f
      cs%f_rnucpe(:,:) = 0._f; cs%f_rhompe(:,:) = 0._f
      cs%f_rnuclg(:,:,:) = 0._f
      rc_loc = 0
      call maxconc(carma, cs, iz, rc_loc); if (rc_loc < 0) rc_loc = 0
      call sulfnuc(carma, cs, iz, rc_loc); if (rc_loc < 0) rc_loc = 0
      call growevapl(carma, cs, iz, rc_loc); if (rc_loc < 0) rc_loc = 0

      ! Snapshot pc before psolve loop
      allocate(pc_snap(carma%f_NBIN, carma%f_NELEM))
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_snap(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      call dump_alloc_2d(prefix, 'pc_prepsolve_probe', pc_snap)
      call dump_alloc_2d(prefix, 'rhompe_probe', cs%f_rhompe)

      ! Run growp, upgxfer, psolve per (ibin, ielem)
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          call growp(carma, cs, iz, ib, ie, rc_loc); if (rc_loc < 0) rc_loc = 0
          call upgxfer(carma, cs, iz, ib, ie, rc_loc); if (rc_loc < 0) rc_loc = 0
          call psolve(carma, cs, iz, ib, ie, rc_loc); if (rc_loc < 0) rc_loc = 0
        end do
      end do

      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_snap(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      call dump_alloc_2d(prefix, 'pc_postpsolve_probe', pc_snap)
      deallocate(pc_snap)

      cs%f_pc = pc_save; cs%f_gc = gc_save; cs%f_t = t_save
      deallocate(pc_save, gc_save, t_save)
    end block
  end subroutine dump_psolve_probe


  !! Probe: compute evaplg via growevapl, then call evapp at end-of-step
  !! state and dump evappe (NBIN, NELEM). Also calls downgevapply and
  !! dumps the post-apply pc (NBIN, NELEM at iz=1).
  subroutine dump_evapp_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: pc_probe(:,:)
    integer                                 :: rc_loc, ie, ib
    integer, parameter                      :: iz = 1
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz; integer, intent(inout) :: rc
      end subroutine maxconc
      subroutine growevapl(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz; integer, intent(inout) :: rc
      end subroutine growevapl
      subroutine evapp(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz; integer, intent(inout) :: rc
      end subroutine evapp
      subroutine downgevapply(carma, cstate, iz, rc)
        use carma_precision_mod; use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz; integer, intent(inout) :: rc
      end subroutine downgevapply
    end interface

    if (.not. allocated(cs%f_evappe)) return
    block
      real(kind=f), allocatable :: pc_save(:,:,:)
      allocate(pc_save(size(cs%f_t), carma%f_NBIN, carma%f_NELEM))
      pc_save = cs%f_pc

      cs%f_growlg(:,:) = 0._f; cs%f_evaplg(:,:) = 0._f
      cs%f_evappe(:,:) = 0._f; cs%f_rnucpe(:,:) = 0._f
      rc_loc = 0
      call maxconc(carma, cs, iz, rc_loc)
      if (rc_loc < 0) rc_loc = 0
      call growevapl(carma, cs, iz, rc_loc)
      if (rc_loc < 0) rc_loc = 0
      call evapp(carma, cs, iz, rc_loc)
      if (rc_loc < 0) rc_loc = 0
      call dump_alloc_2d(prefix, 'evappe_probe', cs%f_evappe)

      allocate(pc_probe(carma%f_NBIN, carma%f_NELEM))
      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_probe(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      call dump_alloc_2d(prefix, 'pc_predowng_probe', pc_probe)

      call downgevapply(carma, cs, iz, rc_loc)
      if (rc_loc < 0) rc_loc = 0

      do ie = 1, carma%f_NELEM
        do ib = 1, carma%f_NBIN
          pc_probe(ib, ie) = cs%f_pc(iz, ib, ie)
        end do
      end do
      call dump_alloc_2d(prefix, 'pc_postdowng_probe', pc_probe)
      deallocate(pc_probe)

      cs%f_pc = pc_save
      deallocate(pc_save)
    end block
  end subroutine dump_evapp_probe


  subroutine dump_carma_ppm(prefix)
    character(len=*), intent(in)         :: prefix
    real(kind=f), allocatable            :: dm2d(:,:), pratt3d(:,:,:), prat3d(:,:,:)
    real(kind=f), allocatable            :: pden1_2d(:,:), palr2d(:,:)
    real(kind=f), allocatable            :: igrowgas_r(:)
    integer                              :: ig, ib, nb, ng, ne

    nb = carma%f_NBIN
    ng = carma%f_NGROUP
    ne = carma%f_NELEM
    allocate(dm2d(nb, ng), pratt3d(3, nb, ng), prat3d(4, nb, ng))
    allocate(pden1_2d(nb, ng), palr2d(4, ng), igrowgas_r(ne))
    ! dm is per-group in carmagroup_type
    do ig = 1, ng
      do ib = 1, nb
        dm2d(ib, ig) = carma%f_group(ig)%f_dm(ib)
      end do
    end do
    ! pratt, prat, pden1, palr are in carma_type directly
    pratt3d = carma%f_pratt
    prat3d  = carma%f_prat
    pden1_2d = carma%f_pden1
    palr2d  = carma%f_palr
    do ig = 1, ne
      igrowgas_r(ig) = real(carma%f_igrowgas(ig), kind=f)
    end do
    call dump_alloc_2d(prefix, 'dm_bin',   dm2d)
    call dump_alloc_3d(prefix, 'pratt',    pratt3d)
    call dump_alloc_3d(prefix, 'prat',     prat3d)
    call dump_alloc_2d(prefix, 'pden1',    pden1_2d)
    call dump_alloc_2d(prefix, 'palr',     palr2d)
    call dump_alloc_1d(prefix, 'igrowgas', igrowgas_r)

    ! Additional static config tables for nsubsteps and other solvers.
    block
      real(kind=f), allocatable :: inucgas_r(:), nnuc2elem_r(:)
      real(kind=f), allocatable :: ienconc_r(:), itype_r(:), igelem_r(:)
      real(kind=f), allocatable :: inuc2elem_r(:,:), inucproc_r(:,:)
      real(kind=f), allocatable :: is_grp_ice_r(:)
      integer :: i, j
      allocate(inucgas_r(ng), nnuc2elem_r(ne), ienconc_r(ng))
      allocate(itype_r(ne), igelem_r(ne), is_grp_ice_r(ng))
      allocate(inuc2elem_r(ne, ne), inucproc_r(ne, ne))
      do i = 1, ng
        inucgas_r(i)   = real(carma%f_inucgas(i),   kind=f)
        ienconc_r(i)   = real(carma%f_group(i)%f_ienconc, kind=f)
        is_grp_ice_r(i) = merge(1._f, 0._f, carma%f_group(i)%f_is_ice)
      end do
      do i = 1, ne
        nnuc2elem_r(i) = real(carma%f_nnuc2elem(i), kind=f)
        itype_r(i)     = real(carma%f_element(i)%f_itype, kind=f)
        igelem_r(i)    = real(carma%f_element(i)%f_igroup, kind=f)
      end do
      do j = 1, ne
        do i = 1, ne
          inuc2elem_r(i, j) = real(carma%f_inuc2elem(i, j), kind=f)
          inucproc_r(i, j)  = real(carma%f_inucproc(i, j),  kind=f)
        end do
      end do
      call dump_alloc_1d(prefix, 'inucgas',   inucgas_r)
      call dump_alloc_1d(prefix, 'nnuc2elem', nnuc2elem_r)
      call dump_alloc_1d(prefix, 'ienconc',   ienconc_r)
      call dump_alloc_1d(prefix, 'itype',     itype_r)
      call dump_alloc_1d(prefix, 'igelem',    igelem_r)
      call dump_alloc_1d(prefix, 'is_grp_ice', is_grp_ice_r)
      call dump_alloc_2d(prefix, 'inuc2elem', inuc2elem_r)
      call dump_alloc_2d(prefix, 'inucproc',  inucproc_r)
      deallocate(inucgas_r, nnuc2elem_r, ienconc_r, itype_r, igelem_r)
      deallocate(is_grp_ice_r, inuc2elem_r, inucproc_r)
    end block

    deallocate(dm2d, pratt3d, prat3d, pden1_2d, palr2d, igrowgas_r)

    ! Coag setup tables (kbin, volx, pkernel, npairl, npairu)
    ! Shapes (Fortran 1-based):
    !   kbin    (NGROUP,NGROUP,NGROUP,NBIN,NBIN)
    !   volx    (NGROUP,NGROUP,NGROUP,NBIN,NBIN)
    !   pkernel (NBIN,NBIN,NGROUP,NGROUP,NGROUP,6)
    !   npairl  (NGROUP,NBIN)
    !   npairu  (NGROUP,NBIN)
    block
      real(kind=f), allocatable :: kbin5(:,:,:,:,:), volx5(:,:,:,:,:)
      real(kind=f), allocatable :: pkernel6(:,:,:,:,:,:)
      real(kind=f), allocatable :: npairl_r(:,:), npairu_r(:,:)
      integer :: i1,i2,i3,i4,i5,i6
      if (allocated(carma%f_kbin)) then
        allocate(kbin5(ng, ng, ng, nb, nb))
        do i5 = 1, nb; do i4 = 1, nb
          do i3 = 1, ng; do i2 = 1, ng; do i1 = 1, ng
            kbin5(i1,i2,i3,i4,i5) = real(carma%f_kbin(i1,i2,i3,i4,i5), kind=f)
          end do; end do; end do
        end do; end do
        call dump_alloc_5d(prefix, 'kbin', kbin5)
        deallocate(kbin5)
      end if
      if (allocated(carma%f_volx)) then
        allocate(volx5(ng, ng, ng, nb, nb))
        volx5 = carma%f_volx
        call dump_alloc_5d(prefix, 'volx', volx5)
        deallocate(volx5)
      end if
      if (allocated(carma%f_pkernel)) then
        allocate(pkernel6(nb, nb, ng, ng, ng, 6))
        pkernel6 = carma%f_pkernel
        call dump_alloc_6d(prefix, 'pkernel', pkernel6)
        deallocate(pkernel6)
      end if
      if (allocated(carma%f_npairl)) then
        allocate(npairl_r(ng, nb), npairu_r(ng, nb))
        do i2 = 1, nb; do i1 = 1, ng
          npairl_r(i1, i2) = real(carma%f_npairl(i1, i2), kind=f)
          npairu_r(i1, i2) = real(carma%f_npairu(i1, i2), kind=f)
        end do; end do
        call dump_alloc_2d(prefix, 'npairl', npairl_r)
        call dump_alloc_2d(prefix, 'npairu', npairu_r)
        deallocate(npairl_r, npairu_r)
      end if
    end block
  end subroutine dump_carma_ppm


  subroutine dump_growevapl_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    integer                                 :: rc_loc
    integer, parameter                      :: iz = 1
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod
        use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz
        integer, intent(inout)               :: rc
      end subroutine maxconc
      subroutine growevapl(carma, cstate, iz, rc)
        use carma_precision_mod
        use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz
        integer, intent(inout)               :: rc
      end subroutine growevapl
    end interface

    if (.not. allocated(cs%f_growlg)) return
    ! Zero growlg/evaplg so probe output is purely from this call.
    cs%f_growlg(:,:) = 0._f
    cs%f_evaplg(:,:) = 0._f
    rc_loc = 0
    call maxconc(carma, cs, iz, rc_loc)
    if (rc_loc < 0) rc_loc = 0
    call growevapl(carma, cs, iz, rc_loc)
    if (rc_loc < 0) rc_loc = 0
    call dump_alloc_2d(prefix, 'growlg_probe', cs%f_growlg)
    call dump_alloc_2d(prefix, 'evaplg_probe', cs%f_evaplg)
  end subroutine dump_growevapl_probe


  subroutine dump_pheat_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    real(kind=f), allocatable               :: dmdt(:)
    integer                                 :: ib, nb, igrp, iep, igas_s
    integer                                 :: rc_loc
    integer, parameter                      :: iz = 1
    interface
      subroutine pheat(carma, cstate, iz, igroup, iepart, ibin, igas, dmdt, rc)
        use carma_precision_mod
        use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz, igroup, iepart, ibin, igas
        real(kind=f), intent(out)            :: dmdt
        integer, intent(inout)               :: rc
      end subroutine pheat
    end interface

    if (.not. allocated(cs%f_gro)) return
    nb      = carma%f_NBIN
    igrp    = 1   ! single group
    iep     = carma%f_group(igrp)%f_ienconc
    igas_s  = carma%f_igash2so4   ! H2SO4 gas index (1-based Fortran)
    if (igas_s == 0) return       ! gas not registered

    ! growevapl calls pheat for ibin=1..NBIN-1 (1-based) only — the last
    ! bin has no upper boundary, so pheat is never called for it.
    ! Probe matches that range and sets dmdt(nb)=0 for the unused slot.
    allocate(dmdt(nb))
    dmdt(:) = 0._f
    rc_loc = 0
    do ib = 1, nb-1
      call pheat(carma, cs, iz, igrp, iep, ib, igas_s, dmdt(ib), rc_loc)
      if (rc_loc < 0) then; dmdt(ib) = 0._f; rc_loc = 0; end if
    end do
    call dump_alloc_1d(prefix, 'pheat_probe', dmdt)
    deallocate(dmdt)
  end subroutine dump_pheat_probe


  subroutine dump_maxconc_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    integer                                 :: rc_loc
    integer, parameter                      :: iz = 1
    interface
      subroutine maxconc(carma, cstate, iz, rc)
        use carma_precision_mod
        use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz
        integer, intent(inout)               :: rc
      end subroutine maxconc
    end interface

    if (.not. allocated(cs%f_pconmax)) return
    rc_loc = 0
    call maxconc(carma, cs, iz, rc_loc)
    call dump_alloc_2d(prefix, 'maxconc_probe', cs%f_pconmax)
  end subroutine dump_maxconc_probe


  subroutine dump_gasexchange_probe(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(inout)    :: cs
    integer                                 :: rc_loc
    integer, parameter                      :: iz = 1
    interface
      subroutine gasexchange(carma, cstate, iz, rc)
        use carma_precision_mod
        use carma_types_mod
        type(carma_type), intent(in)         :: carma
        type(carmastate_type), intent(inout) :: cstate
        integer, intent(in)                  :: iz
        integer, intent(inout)               :: rc
      end subroutine gasexchange
    end interface

    if (.not. allocated(cs%f_gasprod)) return

    cs%f_gasprod(:) = 0._f
    rc_loc = 0
    call gasexchange(carma, cs, iz, rc_loc)
    call dump_alloc_1d(prefix, 'gasexchange_probe', cs%f_gasprod)
  end subroutine dump_gasexchange_probe


  !! Dump the per-bin, per-group dry radius (r) and dry mass (rmass)
  !! from the carma object as 2D arrays of shape (NBIN, NGROUP). These
  !! are static across substeps, so this is called only at step 1.
  !! (rho is per-element, not per-group, so it's not dumped here —
  !! per-cell dry density is in cstate%f_rhop.)
  subroutine dump_carma_bins(prefix)
    character(len=*), intent(in)         :: prefix
    real(kind=f), allocatable            :: r2d(:,:), rmass2d(:,:), rmassup2d(:,:)
    real(kind=f), allocatable            :: rmrat1d(:), rrat2d(:,:), rprat2d(:,:)
    integer                              :: ig, ib, nb, ng

    nb = carma%f_NBIN
    ng = carma%f_NGROUP
    allocate(r2d(nb, ng), rmass2d(nb, ng), rmassup2d(nb, ng), rmrat1d(ng))
    allocate(rrat2d(nb, ng), rprat2d(nb, ng))
    do ig = 1, ng
      do ib = 1, nb
        r2d(ib, ig)       = carma%f_group(ig)%f_r(ib)
        rmass2d(ib, ig)   = carma%f_group(ig)%f_rmass(ib)
        rmassup2d(ib, ig) = carma%f_group(ig)%f_rmassup(ib)
        rrat2d(ib, ig)    = carma%f_group(ig)%f_rrat(ib)
        rprat2d(ib, ig)   = carma%f_group(ig)%f_rprat(ib)
      end do
      rmrat1d(ig) = carma%f_group(ig)%f_rmrat
    end do
    call dump_alloc_2d(prefix, 'r_bin',       r2d)
    call dump_alloc_2d(prefix, 'rmass_bin',   rmass2d)
    call dump_alloc_2d(prefix, 'rmassup_bin', rmassup2d)
    call dump_alloc_1d(prefix, 'rmrat_group', rmrat1d)
    call dump_alloc_2d(prefix, 'rrat',        rrat2d)
    call dump_alloc_2d(prefix, 'rprat',       rprat2d)
    deallocate(r2d, rmass2d, rmassup2d, rmrat1d, rrat2d, rprat2d)
  end subroutine dump_carma_bins


  subroutine dump_totevap(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(in)       :: cs
    real(kind=f), allocatable               :: as_float(:,:)
    integer                                 :: nb, ng, ib, ig
    if (.not. allocated(cs%f_totevap)) return
    nb = size(cs%f_totevap, 1); ng = size(cs%f_totevap, 2)
    allocate(as_float(nb, ng))
    do ig = 1, ng
      do ib = 1, nb
        if (cs%f_totevap(ib, ig)) then
          as_float(ib, ig) = 1._f
        else
          as_float(ib, ig) = 0._f
        end if
      end do
    end do
    call dump_alloc_2d(prefix, 'totevap', as_float)
    deallocate(as_float)
  end subroutine dump_totevap


  subroutine dump_alloc_3d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:,:,:)
    character(len=512)                   :: path
    integer                              :: u, ios
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_3d


  subroutine dump_alloc_5d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:,:,:,:,:)
    character(len=512)                   :: path
    integer                              :: u, ios
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_5d


  subroutine dump_alloc_6d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:,:,:,:,:,:)
    character(len=512)                   :: path
    integer                              :: u, ios
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_6d


  subroutine dump_alloc_2d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:,:)
    character(len=512)                   :: path
    integer                              :: u, ios
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_2d


  !! Per-substep probes: call sulfate_density and sulfate_surf_tens at
  !! the current (wtpct, t) and dump the scalar results. This gives a
  !! direct bench point for the JAX sulfate_density / sulfate_surf_tens
  !! kernels with the exact same inputs Fortran is using.
  subroutine dump_sulfate_probes(prefix, cs)
    character(len=*), intent(in)            :: prefix
    type(carmastate_type), intent(in)       :: cs
    integer                                 :: rc_loc, kk
    real(kind=f), allocatable               :: sulfdens(:), sulfsurf(:)
    integer                                 :: nz_loc

    if (.not. allocated(cs%f_t) .or. .not. allocated(cs%f_wtpct)) return
    nz_loc = size(cs%f_t)
    allocate(sulfdens(nz_loc), sulfsurf(nz_loc))
    rc_loc = 0
    do kk = 1, nz_loc
      sulfdens(kk) = sulfate_density(carma, cs%f_wtpct(kk), cs%f_t(kk), rc_loc)
      sulfsurf(kk) = sulfate_surf_tens(carma, cs%f_wtpct(kk), cs%f_t(kk), rc_loc)
    end do
    call dump_alloc_1d(prefix, 'sulfdens', sulfdens)
    call dump_alloc_1d(prefix, 'sulfsurf', sulfsurf)
    deallocate(sulfdens, sulfsurf)
  end subroutine dump_sulfate_probes


  subroutine dump_alloc_1d(prefix, name, arr)
    character(len=*), intent(in)            :: prefix, name
    real(kind=f), intent(in)                :: arr(:)
    character(len=512)                      :: path
    integer                                 :: u, ios
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_1d


  subroutine dump_3d(prefix, name, arr)
    character(len=*), intent(in)            :: prefix, name
    real(kind=f), allocatable, intent(in)   :: arr(:,:,:)
    character(len=512)                      :: path
    integer                                 :: u, ios
    if (.not. allocated(arr)) return
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_3d


  subroutine dump_1d(prefix, name, arr)
    character(len=*), intent(in)            :: prefix, name
    real(kind=f), allocatable, intent(in)   :: arr(:)
    character(len=512)                      :: path
    integer                                 :: u, ios
    if (.not. allocated(arr)) return
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_1d


  subroutine dump_2d(prefix, name, arr)
    character(len=*), intent(in)            :: prefix, name
    real(kind=f), allocatable, intent(in)   :: arr(:,:)
    character(len=512)                      :: path
    integer                                 :: u, ios
    if (.not. allocated(arr)) return
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_2d


  subroutine dump_5d(prefix, name, arr)
    character(len=*), intent(in)            :: prefix, name
    real(kind=f), allocatable, intent(in)   :: arr(:,:,:,:,:)
    character(len=512)                      :: path
    integer                                 :: u, ios
    if (.not. allocated(arr)) return
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios)
    if (ios /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_5d

end subroutine test_sulfate_diagnostic
