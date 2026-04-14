!! Parameterized CARMA coagulation test.
!! Reads N0, ck0, dtime, nstep from a namelist file (coag_params.nml).
!! Writes output to carma_coagtest_param.txt.

program carma_coagtest_param
  implicit none

  write(*,*) "Parameterized Coagulation Test"
  call test_coagulation()
  write(*,*) "Done"
end program


subroutine test_coagulation()
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

  ! Parameters read from namelist
  real(kind=f) :: param_n0, param_ck0, param_dtime
  integer      :: param_nstep, param_init_bin
  namelist /coag_params/ param_n0, param_ck0, param_dtime, param_nstep, param_init_bin

  real(kind=f), parameter   :: deltaz = 100._f
  real(kind=f), parameter   :: zmin   = 0._f

  type(carma_type), target  :: carma
  type(carma_type), pointer :: carma_ptr
  type(carmastate_type)     :: cstate
  integer                   :: rc = 0

  real(kind=f), allocatable   :: zc(:,:,:)
  real(kind=f), allocatable   :: zl(:,:,:)
  real(kind=f), allocatable   :: p(:,:,:)
  real(kind=f), allocatable   :: pl(:,:,:)
  real(kind=f), allocatable   :: t(:,:,:)
  real(kind=f), allocatable   :: rhoa(:,:,:)

  real(kind=f), allocatable, target  :: mmr(:,:,:,:,:)

  real(kind=f), allocatable          :: lat(:,:)
  real(kind=f), allocatable          :: lon(:,:)

  integer               :: i, j, ix, iy, ixy, istep, ielem, ibin
  integer, parameter    :: lun = 42

  real(kind=f)          :: time
  real(kind=f)          :: rmin, rmrat, rho
  real(kind=f)          :: r(NBIN), dr(NBIN), rmass(NBIN)

  ! Set defaults
  param_n0 = 1.e6_f
  param_ck0 = 8._f * bk * 298._f / 3._f / 1.85e-4_f
  param_dtime = 600._f
  param_nstep = 20
  param_init_bin = 1

  ! Read namelist
  open(unit=10, file="coag_params.nml", status="old", iostat=rc)
  if (rc == 0) then
    read(10, nml=coag_params, iostat=rc)
    close(10)
  end if
  rc = 0

  open(unit=lun, file="carma_coagtest_param.txt", status="unknown")

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
  call CARMA_AddCoagulation(carma, 1, 1, 1, I_COLLEC_DATA, rc, ck0=param_ck0)
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

  ! Initial conditions: N0 particles in specified bin
  mmr(:,:,:,:,:) = 0._f
  call CARMAGROUP_Get(carma, 1, rc, rmass=rmass, r=r, dr=dr)

  mmr(1,:,:,1,param_init_bin) = rmass(param_init_bin)/1000._f * param_n0 * 1e6_f &
                               / (p(1,:,:)/287._f/t(1,:,:))

  ! Write header
  write(lun,*) NBIN, NELEM, NGROUP

  do i = 1, NBIN
    write(lun,'(i3,2(1x,e12.5))') i, r(i), dr(i)
  end do

  ! Write initial state
  write(lun,*) 0
  rhoa(:,:,:) = p(:,:,:)/287._f/t(:,:,:)
  do j = 1, NELEM
    do i = 1, NBIN
      write(lun,'(i3,1(1x,e12.5))') i, &
        mmr(1,NY,NX,j,i)*rhoa(1,NY,NX)/rmass(i)*1e-6_f*1e3_f
    end do
  end do

  ! Time integration
  do istep = 1, param_nstep
    time = (istep - 1) * param_dtime

    do ixy = 1, NX*NY
      ix = ((ixy-1) / NY) + 1
      iy = ixy - (ix-1)*NY

      call CARMASTATE_Create(cstate, carma_ptr, time, param_dtime, NZ, &
                             I_CART, lat(iy,ix), lon(iy,ix), &
                             zc(:,iy,ix), zl(:,iy,ix), p(:,iy,ix), &
                             pl(:,iy,ix), t(:,iy,ix), rc)

      do ielem = 1, NELEM
        do ibin = 1, NBIN
          call CARMASTATE_SetBin(cstate, ielem, ibin, mmr(:,iy,ix,ielem,ibin), rc)
        end do
      end do

      call CARMASTATE_Step(cstate, rc)

      do ielem = 1, NELEM
        do ibin = 1, NBIN
          call CARMASTATE_GetBin(cstate, ielem, ibin, mmr(:,iy,ix,ielem,ibin), rc)
        end do
      end do

      call CARMASTATE_GetState(cstate, rc, t=t(:,iy,ix))
    enddo

    ! Write timestep output
    write(lun,'(f12.1)') istep*param_dtime
    rhoa(:,:,:) = p(:,:,:)/287._f/t(:,:,:)
    do j = 1, NELEM
      do i = 1, NBIN
        write(lun,'(i3,1(1x,e12.5))') i, &
          mmr(1,NY,NX,j,i)*rhoa(1,NY,NX)/rmass(i)*1e-6_f*1e3_f
      end do
    end do
  end do

  call CARMASTATE_Destroy(cstate, rc)
  close(unit=lun)
  call CARMA_Destroy(carma, rc)

end subroutine
