build_lib_example(
  NAME interference-pattern
  SOURCE_FILES interference-pattern/mygym.cc
               interference-pattern/sim.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libinternet}
    ${libopengym}
    ${libwifi}
)

build_lib_example(
  NAME linear-mesh
  SOURCE_FILES linear-mesh/sim.cc
  LIBRARIES_TO_LINK
    ${libapplications}
    ${libcore}
    ${libinternet}
    ${libopengym}
    ${libwifi}
)

build_lib_example(
  NAME linear-mesh-2
  SOURCE_FILES linear-mesh-2/sim.cc
               linear-mesh-2/mygym.cc
  LIBRARIES_TO_LINK
    ${libapplications}
    ${libcore}
    ${libinternet}
    ${libopengym}
    ${libwifi}
)


build_lib_example(
  NAME MLORLWork
  SOURCE_FILES PolicyGradient_MLO/sim.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libopengym}
    ${libwifi}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libnetwork}
    ${libstats}
    ${libspectrum}
    ${libflow-monitor}
    ${libpropagation}
)


build_lib_example(
  NAME opengym
  SOURCE_FILES opengym/sim.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libopengym}
    ${libwifi}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libnetwork}
    ${libstats}
    ${libspectrum}
    ${libflow-monitor}
    ${libpropagation}
)

build_lib_example(
  NAME opengym-2
  SOURCE_FILES
    opengym-2/sim.cc
    opengym-2/mygym.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libwifi}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libpropagation}
    ${libopengym}
)

build_lib_example(
  NAME rl-tcp
  SOURCE_FILES rl-tcp/sim.cc
               rl-tcp/tcp-rl-env.cc
               rl-tcp/tcp-rl.cc
  LIBRARIES_TO_LINK
    ${libapplications}
    ${libcore}
    ${libflow-monitor}
    ${libinternet}
    ${libopengym}
    ${libpoint-to-point-layout}
    ${libptp}
)

# Additional linking for interference-pattern example
if(TARGET interference-pattern)
    target_link_libraries(interference-pattern
        ${libpoint-to-point}
        ${libapplications}
        ${libflow-monitor}
        ${libinternet}
        ${libspectrum}
        ${libwifi}
        ${libmobility}
        ${libstats}
    )
    
endif()
