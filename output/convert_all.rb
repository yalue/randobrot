# This script converts all .ppm images to .png images. It just calls the
# "convert" command-line tool.
files = Dir["*.ppm"]
files.each do |f|
  `convert "#{f}" "#{File.basename(f, ".ppm")}.png"`
  File.delete(f)
end

