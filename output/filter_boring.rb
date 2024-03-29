# This uses chunky_png to detect mostly-white or mostly-black images and move
# them elsewhere (typically to be deleted). For this to work, the images must
# have already been converted to PNGs, such as by running convert_all.rb.
require 'chunky_png'

def get_brightness(c)
  ChunkyPNG::Color.r(ChunkyPNG::Color.to_grayscale(c))
end

# Returns true if the given picture is over 90% black or over 90% white.
def is_boring(pic)
  total_count = pic.height * pic.width
  white_count = 0
  black_count = 0
  pic.height.times do |y|
    pic.width.times do |x|
      v = get_brightness(pic[x, y])
      white_count += 1 if v >= 0xfd
      black_count += 1 if v <= 2
    end
  end
  too_white = (white_count.to_f / total_count.to_f) > 0.9
  too_black = (black_count.to_f / total_count.to_f) > 0.9
  too_black || too_white
end

def process_file_array(files)
  files.each do |f|
    pic = ChunkyPNG::Image.from_file(f)
    base_name = File.basename(f)
    if is_boring(pic)
      puts "#{base_name} is boring"
      #`mv "#{base_name}" "./boring_pics/#{base_name}"`
      `rm "#{base_name}"`
    else
      puts "#{base_name} is OK"
    end
  end
end

files = Dir["./*.png"]
threads = []
files.each_slice(files.size / 10) do |array|
  threads << Thread.new do
    process_file_array(array)
  end
end
puts "Waiting for threads to finish..."
threads.each {|t| t.join()}

