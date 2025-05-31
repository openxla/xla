project('xla', 'cpp', default_options : ['default_library=shared'])

if not meson.is_subproject() 
  cc = meson.get_compiler('cpp')
endif

include = include_directories('.')

xla = library('xla',
  include_directories : [include],
  install : true,
  link_args : [],
)

xla_dep = declare_dependency(
  link_with : xla,
  include_directories : [include]
)
