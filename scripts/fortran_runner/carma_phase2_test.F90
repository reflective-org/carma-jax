!! Phase 2 validation: computed kernel coagulation test (47 bins).
!! Uses computed Brownian+gravitational kernel (no ck0 constant).
!! Reads per-bin initial concentrations from init_nd.txt.
!! Writes intermediate values (vf, ckernel) and final pc.
!! Output: carma_phase2_out.txt

program carma_phase2_test
  implicit none
  write(*,*) "Phase 2 Validation Test"
  call test_phase2()
  write(*,*) "Done"
end program


subroutine test_phase2()
  use carma_precision_mod
  use carma_constants_mod
  use carma_enums_mod
  use carma_types_mod
  use carmaelement_mod
  use carmagroup_mod
  use carmastate_mod
  use carma_mod
  use atmosphere_mod

  implicit none

  integer, parameter    :: NX           = 1
  integer, parameter    :: NY           = 1
  integer, parameter    :: NZ           = 1
  integer, parameter    :: NZP1         = NZ+1
  integer, parameter    :: NELEM        = 1
  integer, parameter    :: NBIN         = 47
  integer, parameter    :: NGROUP       = 1
  integer, parameter    :: NSOLUTE      = 0
  integer, parameter    :: NGAS         = 0
  integer, parameter    :: NWAVE        = 0

  integer, parameter        :: I_DUST       = 1

  real(kind=f) :: param_dtime
  integer      :: param_nstep
  namelist /phase2_params/ param_dtime, param_nstep

  real(kind=f), parameter   :: deltaz = 100._f
  real(kind=f), parameter   :: zmin   = 0._f

  type(carma_type), target  :: carma
  type(carma_type), pointer :: carma_ptr
  type(carmastate_type)     :: cstate
  integer                   :: rc = 0

  real(kind=f), allocatable   :: zc(:,:,:), zl(:,:,:), p(:,:,:), pl(:,:,:)
  real(kind=f), allocatable   :: t(:,:,:), rhoa(:,:,:)
  real(kind=f), allocatable, target  :: mmr(:,:,:,:,:)
  real(kind=f), allocatable          :: lat(:,:), lon(:,:)

  integer               :: i, j, ix, iy, ixy, istep, ielem, ibin
  integer, parameter    :: lun = 42

  real(kind=f)          :: time
  real(kind=f)          :: rmin, rmrat, rho
  real(kind=f)          :: r(NBIN), dr(NBIN), rmass(NBIN)
  real(kind=f)          :: vf_out(NBIN)

  ! Per-bin initial number concentrations
  real(kind=f)          :: init_nd(NBIN)

  ! Defaults
  param_dtime = 60._f
  param_nstep = 1

  ! Read params
  open(unit=10, file="phase2_params.nml", status="old", iostat=rc)
  if (rc == 0) then
    read(10, nml=phase2_params, iostat=rc)
    close(10)
  end if
  rc = 0

  ! Read per-bin initial concentrations
  init_nd(:) = 0._f
  open(unit=11, file="init_nd.txt", status="old", iostat=rc)
  if (rc == 0) then
    do i = 1, NBIN
      read(11, *, iostat=rc) init_nd(i)
      if (rc /= 0) exit
    end do
    close(11)
    rc = 0
  end if

  open(unit=lun, file="carma_phase2_out.txt", status="unknown")

  allocate(zc(NZ,NY,NX), zl(NZP1,NY,NX), p(NZ,NY,NX), pl(NZP1,NY,NX), &
           t(NZ,NY,NX), rhoa(NZ,NY,NX))
  allocate(mmr(NZ,NY,NX,NELEM,NBIN))
  allocate(lat(NY,NX), lon(NY,NX))

  call CARMA_Create(carma, NBIN, NELEM, NGROUP, NSOLUTE, NGAS, NWAVE, rc, LUNOPRT=6)
  carma_ptr => carma

  rho = 2._f
  rmrat = 2._f
  rmin = 2.e-8_f

  call CARMAGROUP_Create(carma, 1, 'aerosol', rmin, rmrat, I_SPHERE, 1._f, .FALSE., rc)
  call CARMAELEMENT_Create(carma, 1, 1, "dust", rho, I_INVOLATILE, I_DUST, rc)

  ! Computed kernel with Fuchs collection efficiency (no ck0)
  call CARMA_AddCoagulation(carma, 1, 1, 1, I_COLLEC_FUCHS, rc)

  call CARMA_Initialize(carma, rc, do_coag=.TRUE.)

  lat(:,:) = 40.0_f
  lon(:,:) = -105.0_f

  do i = 1, NZ
    zc(i,:,:) = zmin + (deltaz * (i - 0.5_f))
  end do
  call GetStandardAtmosphere(zc, p=p, t=t)

  do i = 1, NZP1
    zl(i,:,:) = zmin + ((i - 1) * deltaz)
  end do
  call GetStandardAtmosphere(zl, p=pl)

  call CARMAGROUP_Get(carma, 1, rc, rmass=rmass, r=r, dr=dr)

  ! Set initial conditions
  mmr(:,:,:,:,:) = 0._f
  do ibin = 1, NBIN
    if (init_nd(ibin) > 0._f) then
      mmr(1,:,:,1,ibin) = rmass(ibin)/1000._f * init_nd(ibin) * 1e6_f &
                         / (p(1,:,:)/287._f/t(1,:,:))
    end if
  end do

  ! Write header
  write(lun,'(i4,1x,i6,1x,e15.8,1x,e15.8)') NBIN, param_nstep, t(1,1,1), p(1,1,1)

  ! Write bin radii
  do i = 1, NBIN
    write(lun,'(i3,1x,e15.8)') i, r(i)
  end do

  ! Run simulation
  do istep = 1, param_nstep
    time = (istep - 1) * param_dtime

    call CARMASTATE_Create(cstate, carma_ptr, time, param_dtime, NZ, &
                           I_CART, lat(1,1), lon(1,1), &
                           zc(:,1,1), zl(:,1,1), p(:,1,1), &
                           pl(:,1,1), t(:,1,1), rc)

    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_SetBin(cstate, ielem, ibin, mmr(:,1,1,ielem,ibin), rc)
      end do
    end do

    call CARMASTATE_Step(cstate, rc)

    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_GetBin(cstate, ielem, ibin, mmr(:,1,1,ielem,ibin), rc)
      end do
    end do

    call CARMASTATE_GetState(cstate, rc, t=t(:,1,1))
  enddo

  ! Write final state: number density per bin
  rhoa(1,:,:) = p(1,:,:)/287._f/t(1,:,:)
  do i = 1, NBIN
    write(lun,'(i3,1x,e15.8)') i, &
      mmr(1,1,1,1,i)*rhoa(1,1,1)/rmass(i)*1e-6_f*1e3_f
  end do

  call CARMASTATE_Destroy(cstate, rc)
  close(unit=lun)
  call CARMA_Destroy(carma, rc)

end subroutine
