
import sys

def check_file(path):
    print(f"Checking {path}...")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        lines = content.splitlines()
        
        issues = []
        
        # Check for NaN/Inf
        for i, line in enumerate(lines):
            if 'nan' in line.lower():
                issues.append(f"Line {i+1}: contains NaN: {line.strip()}")
            if 'inf' in line.lower() and 'infinity' not in line.lower(): # exclude "infinity" text if any
                 issues.append(f"Line {i+1}: contains Inf: {line.strip()}")

        print(f"Found {len(issues)} text issues (NaN/Inf).")
        for i in issues[:10]:
            print(i)
            
        import ezdxf
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        print(f"Entities: {len(msp)}")
        
        # Check extents
        extmin = doc.header.get('$EXTMIN', None)
        extmax = doc.header.get('$EXTMAX', None)
        print(f"EXTMIN: {extmin}")
        print(f"EXTMAX: {extmax}")
        
        # Check for 0 length lines
        zero_len = 0
        lines_count = 0
        for e in msp:
            if e.dxftype() == 'LINE':
                lines_count += 1
                start = e.dxf.start
                end = e.dxf.end
                l = ((start[0]-end[0])**2 + (start[1]-end[1])**2)**0.5
                if l < 1e-9:
                    zero_len += 1
        
        print(f"Total Lines: {lines_count}")
        print(f"Zero Length Lines: {zero_len}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_file("model12.dxf")
