import csv

data = [[[[667.0, 21.0], [752.0, 16.0], [753.0, 37.0], [668.0, 42.0]], ('Email Id', 0.9719435572624207)],
 [[[41.0, 35.0], [108.0, 35.0], [108.0, 56.0], [41.0, 56.0]], ('Sr.No.', 0.9919988512992859)],
 [[[144.0, 36.0], [219.0, 36.0], [219.0, 54.0], [144.0, 54.0]], ('Roll No.', 0.9563426971435547)],
 [[[357.0, 35.0], [415.0, 35.0], [415.0, 53.0], [357.0, 53.0]], ('Name', 0.9989587664604187)],
 [[[128.0, 71.0], [227.0, 71.0], [227.0, 85.0], [128.0, 85.0]], ('2110993771', 0.9993621110916138)],
 [[[245.0, 68.0], [349.0, 67.0], [350.0, 85.0], [245.0, 86.0]], ('Baashi Nazir', 0.9490595459938049)],
 [[[527.0, 62.0], [801.0, 49.0], [802.0, 66.0], [528.0, 79.0]], ('baashi3771.be21@,chitkara.edu.in', 0.9790962338447571)],
 [[[65.0, 72.0], [79.0, 72.0], [79.0, 87.0], [65.0, 87.0]], ('1', 0.9962843060493469)],
 [[[60.0, 99.0], [78.0, 99.0], [78.0, 117.0], [60.0, 117.0]], ('2', 0.9971582889556885)],
 [[[125.0, 97.0], [227.0, 96.0], [227.0, 114.0], [126.0, 115.0]], ('2110993795', 0.9986836314201355)],
 [[[243.0, 97.0], [313.0, 97.0], [313.0, 114.0], [243.0, 114.0]], ('Hitakshi', 0.9968388676643372)],
 [[[527.0, 90.0], [815.0, 78.0], [816.0, 95.0], [528.0, 107.0]], ('hitakshi3795.be21@chitkara.edu.in', 0.996421217918396)],
 [[[57.0, 127.0], [76.0, 127.0], [76.0, 147.0], [57.0, 147.0]], ('3', 0.9977352619171143)],
 [[[122.0, 126.0], [224.0, 125.0], [224.0, 143.0], [123.0, 144.0]], ('2110993811', 0.9983865022659302)],
 [[[241.0, 125.0], [338.0, 124.0], [339.0, 142.0], [241.0, 143.0]], ('Mehakpreet', 0.9965227246284485)],
 [[[526.0, 119.0], [847.0, 106.0], [848.0, 124.0], [527.0, 137.0]], ('mehakpreet3811.be21@chitkara.edu.in', 0.9948550462722778)],
 [[[120.0, 155.0], [223.0, 154.0], [223.0, 172.0], [121.0, 173.0]], ('2110993832', 0.9989520311355591)],
 [[[241.0, 154.0], [364.0, 154.0], [364.0, 171.0], [241.0, 171.0]], ('Shivam Pandey', 0.9871168732643127)],
 [[[528.0, 147.0], [811.0, 138.0], [812.0, 156.0], [528.0, 165.0]], ('shivam3832.be21@chitkara.edu.in', 0.9962336421012878)],
 [[[56.0, 157.0], [75.0, 157.0], [75.0, 175.0], [56.0, 175.0]], ('4', 0.9989027976989746)],
 [[[118.0, 184.0], [221.0, 183.0], [221.0, 201.0], [119.0, 202.0]], ('2110993839', 0.9985971450805664)],
 [[[238.0, 183.0], [386.0, 180.0], [386.0, 197.0], [238.0, 200.0]], ('SOHIL DHIMAN', 0.9720785021781921)],
 [[[527.0, 176.0], [795.0, 168.0], [796.0, 186.0], [527.0, 194.0]], ('sohil3839.be21@chitkara.edu.in', 0.9958187937736511)],
 [[[53.0, 186.0], [71.0, 186.0], [71.0, 206.0], [53.0, 206.0]], ('5', 0.9975481629371643)],
 [[[115.0, 214.0], [219.0, 212.0], [219.0, 230.0], [116.0, 232.0]], ('2110993852', 0.9989304542541504)],
 [[[237.0, 213.0], [343.0, 210.0], [344.0, 227.0], [237.0, 230.0]], ('Vrinda Vritti', 0.9624035358428955)],
 [[[528.0, 205.0], [809.0, 198.0], [810.0, 216.0], [528.0, 223.0]], ('vrinda3852.be21@chitkara.edu.in', 0.9960778951644897)],
 [[[51.0, 217.0], [69.0, 217.0], [69.0, 236.0], [51.0, 236.0]], ('6', 0.9793912172317505)],
 [[[236.0, 242.0], [486.0, 237.0], [486.0, 254.0], [236.0, 259.0]], ('ADVITIYA BHARTI GUPTA', 0.9550018310546875)],
 [[[526.0, 234.0], [825.0, 226.0], [826.0, 246.0], [526.0, 254.0]], ('advitiya3858.be21@chitkara.edu.in', 0.9837824702262878)],
 [[[49.0, 245.0], [67.0, 245.0], [67.0, 265.0], [49.0, 265.0]], ('7', 0.9890981912612915)],
 [[[113.0, 244.0], [218.0, 242.0], [218.0, 260.0], [114.0, 262.0]], ('2110993858', 0.9987415075302124)],
 [[[111.0, 274.0], [216.0, 273.0], [216.0, 291.0], [112.0, 292.0]], ('2110993876', 0.9986053705215454)],
 [[[231.0, 272.0], [273.0, 272.0], [273.0, 290.0], [231.0, 290.0]], ('Jatin', 0.996871829032898)],
 [[[526.0, 266.0], [796.0, 259.0], [797.0, 277.0], [526.0, 284.0]], ('jatin3876.be21@chitkara.edu.in', 0.9955425262451172)],
 [[[46.0, 277.0], [64.0, 277.0], [64.0, 296.0], [46.0, 296.0]], ('8', 0.9957637786865234)]]

# Sort data by y-coordinate of first point
data.sort(key=lambda x: x[0][0][1])

# Create CSV file
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Text', 'Confidence'])
    for row in data:
        writer.writerow([row[1][0], row[1][1]])