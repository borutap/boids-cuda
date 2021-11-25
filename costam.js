//BOIDS
//(c) 2010 by Tomasz Lubinski
//www.algorytm.org

var boids = new Array(20);
var r = 50.0;
var degree = 120.0;
var d_min = 20.0;
var weigth_v = 0.1;
var weigth_d = 0.15;
var weigth_min = 0.15;
var weigth_p = 0.1;
var v_max = 4.0;

var width = 600;
var height = 400;

//calculate new speed and direction
function modify_speed_and_direction() {
  var dist = 0.0;
  var deg = 0.0;

  for (i = 0; i < boids.length; i++) {
    boids[i].alg_mean_vx = boids[i].alg_vx;
    boids[i].alg_mean_vy = boids[i].alg_vy;
    boids[i].alg_mean_d = 0;
    boids[i].alg_num = 1;
    for (j = 0; j < boids.length; j++) {
      if (j == i) continue;
      dist = Math.sqrt(
        Math.pow(boids[i].alg_x - boids[j].alg_x, 2) + Math.pow(boids[i].alg_y - boids[j].alg_y, 2)
      );
      deg = Math.acos(
        (boids[i].alg_vx /
          Math.sqrt(boids[i].alg_vx * boids[i].alg_vx + boids[i].alg_vy * boids[i].alg_vy)) *
          ((boids[j].alg_x - boids[i].alg_x) / dist) +
          (boids[i].alg_vy /
            Math.sqrt(boids[i].alg_vx * boids[i].alg_vx + boids[i].alg_vy * boids[i].alg_vy)) *
            ((boids[j].alg_y - boids[i].alg_y) / dist)
      );
      deg = Math.abs((180 * deg) / Math.PI);
      if (dist < r && deg < degree) {
        boids[i].alg_num++;
        boids[i].alg_mean_vx += boids[j].alg_vx;
        boids[i].alg_mean_vy += boids[j].alg_vy;
        boids[i].alg_mean_d += dist;
      }
    }
  }

  for (i = 0; i < boids.length; i++) {
    //adjust speed to neighbours speed
    boids[i].alg_vx += weigth_v * (boids[i].alg_mean_vx / boids[i].alg_num - boids[i].alg_vx);
    boids[i].alg_vy += weigth_v * (boids[i].alg_mean_vy / boids[i].alg_num - boids[i].alg_vy);

    //pertubation
    boids[i].alg_vx += weigth_p * ((Math.random() - 0.5) * v_max);
    boids[i].alg_vy += weigth_p * ((Math.random() - 0.5) * v_max);

    if (boids[i].alg_num > 1) boids[i].alg_mean_d /= boids[i].alg_num - 1;
    for (j = 0; j < boids.length; j++) {
      if (j == i) continue;
      dist = Math.sqrt(
        Math.pow(boids[i].alg_x - boids[j].alg_x, 2) + Math.pow(boids[i].alg_y - boids[j].alg_y, 2)
      );
      deg = Math.acos(
        (boids[i].alg_vx /
          Math.sqrt(boids[i].alg_vx * boids[i].alg_vx + boids[i].alg_vy * boids[i].alg_vy)) *
          ((boids[j].alg_x - boids[i].alg_x) / dist) +
          (boids[i].alg_vy /
            Math.sqrt(boids[i].alg_vx * boids[i].alg_vx + boids[i].alg_vy * boids[i].alg_vy)) *
            ((boids[j].alg_y - boids[i].alg_y) / dist)
      );
      deg = Math.abs((180 * deg) / Math.PI);
      if (dist < r && deg < degree) {
        if (Math.abs(boids[j].alg_x - boids[i].alg_x) > d_min) {
          boids[i].alg_vx +=
            (weigth_d / boids[i].alg_num) *
            (((boids[j].alg_x - boids[i].alg_x) * (dist - boids[i].alg_mean_d)) / dist);
          boids[i].alg_vy +=
            (weigth_d / boids[i].alg_num) *
            (((boids[j].alg_y - boids[i].alg_y) * (dist - boids[i].alg_mean_d)) / dist);
        } //neighbours are too close
        else {
          boids[i].alg_vx -=
            (weigth_min / boids[i].alg_num) *
            (((boids[j].alg_x - boids[i].alg_x) * d_min) / dist -
              (boids[j].alg_x - boids[i].alg_x));
          boids[i].alg_vy -=
            (weigth_min / boids[i].alg_num) *
            (((boids[j].alg_y - boids[i].alg_y) * d_min) / dist -
              (boids[j].alg_y - boids[i].alg_y));
        }
      }
    }

    //check speed is not too high
    if (Math.sqrt(boids[i].alg_vx * boids[i].alg_vx + boids[i].alg_vy * boids[i].alg_vy) > v_max) {
      boids[i].alg_vx *= 0.75;
      boids[i].alg_vy *= 0.75;
    }
  }
}

//move and displat boids
function move_and_display() {
  //first modify speed and direction
  modify_speed_and_direction();

  for (i = 0; i < boids.length; i++) {
    //move boid
    boids[i].alg_x += boids[i].alg_vx;
    boids[i].alg_y += boids[i].alg_vy;

    //check if outside window
    if (boids[i].alg_x > width) boids[i].alg_x -= width;
    else if (boids[i].alg_x < 0) boids[i].alg_x += width;
    if (boids[i].alg_y > height) boids[i].alg_y -= height;
    else if (boids[i].alg_y < 0) boids[i].alg_y += height;

    //display new position of boid
    boids[i].attr({
      path: "M" + boids[i].alg_x + " " + boids[i].alg_y + "l0 8m0 -1 l-3 -7 l6 0l-3 7",
    });
    if (boids[i].alg_vx < 0) {
      boids[i].rotate(
        90.0 + (Math.atan(boids[i].alg_vy / boids[i].alg_vx) * 180.0) / Math.PI,
        true
      );
    } else {
      boids[i].rotate(
        -90.0 + (Math.atan(boids[i].alg_vy / boids[i].alg_vx) * 180.0) / Math.PI,
        true
      );
    }
  }
  setTimeout("move_and_display()", 20);
}

//initialize data
function start() {       
  var paper = Raphael("canvas", width, height);
  var background = paper.rect(0, 0, width, height);
  background.attr({ fill: "#3490c9" });
  paper.text(width - 70, height - 20, "www.algorytm.org");

  for (i = 0; i < boids.length; i++) {
    boids[i] = paper.path("M0 0l0 8m0 -1 l-3 -7 l6 0l-3 7");
    boids[i].alg_x = Math.floor(Math.random() * width);
    boids[i].alg_y = Math.floor(Math.random() * height);
    boids[i].alg_vx = Math.random() * 4.0 - 2.0;
    boids[i].alg_vy = Math.random() * 4.0 - 2.0;
  }

  setTimeout("move_and_display()", 100);
}
