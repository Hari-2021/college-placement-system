import matplotlib.pyplot as plt
import pymysql
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='placement'
)

cursor = connection.cursor()

query = "SELECT company, COUNT(*) as num_applications FROM applied GROUP BY company"
cursor.execute(query)

results = cursor.fetchall()

companies = [result[0] for result in results]
num_applications = [result[1] for result in results]

cursor.close()
connection.close()

# Create a bar chart
plt.bar(companies, num_applications, color='skyblue')
plt.xlabel('Company')
plt.ylabel('Number of Applications')
plt.title('Number of Applications for IT Companies')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

# Save or show the chart
plt.savefig('applications_chart.png')
plt.show()
