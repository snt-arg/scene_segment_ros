#include <vector>
#include <set>
#include <iostream>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include "ros/ros.h"

void pcdCallback(sensor_msgs::PointCloud2ConstPtr const &msg) {
  sensor_msgs::PointCloud2ConstIterator<float> iter_class_prob(*msg, "prob");
  uint32_t width = msg->width;
  uint32_t height = msg->height;
  uint32_t point_step = msg->point_step;
  uint32_t row_step = msg->row_step;
  std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Point step: " << point_step << std::endl;
    std::cout << "Row step: " << row_step << std::endl;
    std::cout << "Start processing probabilities..." << std::endl;
  // INFER IT BY ASSUMING THAT PROBS ARE FLOAT32 AND DOING POINT_STEP / 4
  // ANOTHER OPTION IS TO INCLUDE IT IN THE PCD MESSAGE OR AS A PARAMETER
  int num_classes = point_step / 4;
  std::cout << "Number of classes: " << num_classes << std::endl;
  
  std::vector<std::vector<std::vector<float>>> prob_mat(
      height, std::vector<std::vector<float>>(
                  width, std::vector<float>(num_classes, 0.0)));
  
    // FILL PROBABILITY MATRIX
    int counter = 0;
    int curr_row = 0;
    int curr_col = 0;
    for(; iter_class_prob != iter_class_prob.end(); ++iter_class_prob) {
        for(uint32_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            // std::cout << "Counter: " << counter << " Row: " << curr_row << " Col: " << curr_col << " Class: " << class_idx << " Prob: " << iter_class_prob[class_idx] << std::endl;
            prob_mat[curr_row][curr_col][class_idx] = iter_class_prob[class_idx];
        }
        counter++;
        curr_row++;
        if (curr_row == height) {
            curr_row = 0;
            curr_col++;
        }
    }

    // TO TEST LETS GET THE UNIQUE CLASS IDS FOR MOST PROBABLE CLASS
    std::set<int> unique_class_ids;
    for (uint32_t row = 0; row < height; ++row) {
      for (uint32_t col = 0; col < width; ++col) {
        float max_prob = 0.0;
        int max_class_idx = -1;
        for (uint32_t class_idx = 0; class_idx < num_classes; ++class_idx) {
          if (prob_mat[row][col][class_idx] > max_prob) {
            max_prob = prob_mat[row][col][class_idx];
            max_class_idx = class_idx;
          }
        }
        unique_class_ids.insert(max_class_idx);
      }
    }

    std::cout << "Unique class ids: ";
    for (auto class_id : unique_class_ids) {
      std::cout << class_id << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "pcd_listener");

  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("/camera/color/image_segment/probabilities", 1000, pcdCallback);

  ros::spin();

  return 0;
}