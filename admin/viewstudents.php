<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <!-- <link rel="stylesheet" type="text/css" href="css/addcomp.css"> -->
    <?php include_once 'includes/head.php' ?>

</head>
<body>

      <?php include_once 'includes/nav.php' ?>
    <div class="container" style="z-index: 2;">
      <h1 class="form-row justify-content-center">View Students</h1> <br>
       <form>
     <!-- <img src="images/1.jpg"> -->
     <div class="table-responsive">
    <table class="table table-hover table-borderless table-dark" style="background: #514DC0; border-radius:15px;">
      <thead>
        <tr>
          <th scope="col">Student Name</th>
          <th scope="col">Phone</th>
          <th scope="col">Email</th>
          <th scope="col">Username</th>
          <th scope="col">Course</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <?php
            $sql = "select * from studentlogin;";
            $res = mysqli_query($conn, $sql);
            $rescheck = mysqli_num_rows($res);
            if($rescheck > 0) {
              while ($row = mysqli_fetch_assoc($res)) {
                  echo '<tr>';
                    echo '<td>'.$row['fname'].'</td>';
                    echo '<td>'.$row['phone'].'</td>';
                    echo '<td>'.$row['email'].'</td>';
                    echo '<td>'.$row['uname'].'</td>';
                    echo '<td>'.$row['course'].'</td>';
                    
                  echo '</tr>';
              }
            }
           ?>
      </tbody>
    </table>
    </div>

   </form>
    </div>
    <?php include_once 'includes/footer.php' ?>
</body>
</html>