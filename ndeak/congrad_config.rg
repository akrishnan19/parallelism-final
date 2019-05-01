import "regent"

local c = regentlib.c

local util = {}

struct ConGradConfig
{
  dump_output    : bool,
  error_bound    : double,
  max_iterations : uint32,
  matrix_order   : uint64,
  nnz            : uint64;
}

local cstring = terralib.includec("string.h")

terra print_usage_and_abort()
  c.printf("Usage: regent.py congrad.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h            : Print the usage and exit.\n")
--  c.printf("  -i {file}     : Use {file} as input.\n")
--  c.printf("  -o {file}     : Save the ranks of pages to {file}.\n")
--  c.printf("  -d {value}    : Set the damping factor to {value}.\n")
  c.printf("  -e {value}    : Set the error bound to {value}.\n")
  c.printf("  -c {value}    : Set the maximum number of iterations to {value}.\n")
  c.printf("  -n {value}    : Set the order of the matrix.\n")
  c.exit(0)
end

terra ConGradConfig:initialize_from_command()

  self.error_bound = 1e-5
  self.max_iterations = 1000
  self.matrix_order = 100

  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-h") == 0 then
      print_usage_and_abort()
    elseif cstring.strcmp(args.argv[i], "-e") == 0 then
      i = i + 1
      self.error_bound = c.atof(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-c") == 0 then
      i = i + 1
      self.max_iterations = c.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-n") == 0 then
      i = i + 1
      self.matrix_order = c.atoi(args.argv[i])
    end
    i = i + 1
  end

  self.nnz = ((self.matrix_order - 2) * 3) + 4
end

return ConGradConfig
