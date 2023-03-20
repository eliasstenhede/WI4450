import os
if __name__ == "__main__":
    data = {}
    data_folder = './benchmark_data'
    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), 'r', encoding='ISO-8859-1') as f:
            contents = f.read()
            if 'Gbyte/s' in contents:  # format 1
                lines = contents.split('\n')
                for linenum in range(4,8):
                    test_name = lines[linenum].split()[0].strip()
                    threads = int(lines[0].split()[-1].strip())
                    bandwidth = float(lines[linenum].split()[6].strip())*1024 #GB to MB
                    if test_name not in data:
                        data[test_name] = {'threads': [], 'bandwidth': []}
                    data[test_name]['threads'].append(int(threads))
                    data[test_name]['bandwidth'].append(bandwidth)
            else:  # format 2
                lines = contents.split('\n')
                test_name = ''
                threads = ''
                bandwidth = ''
                addn = False
                for line in lines:
                    if line.startswith('Test:'):
                        test_name = line.strip().split(':')[-1].strip()
                        if test_name not in data:
                            data[test_name] = {'threads': [], 'bandwidth': []}
                    elif line.startswith('Using') and 'threads' in line:
                        threads = line.strip().split()[-2]
                    elif line.startswith('MByte/s:'):
                        bandwidth = line.strip().split()[-1]
                        addn = True
                    if test_name in data and addn:
                        data[test_name]['threads'].append(int(threads))
                        data[test_name]['bandwidth'].append(bandwidth)
                        addn = False
                                              
    print(data)
