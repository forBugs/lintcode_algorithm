package com.buct.pageTransform;

import com.buct.util.ImageViewer;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Administrator on 2017/7/1.
 */
public class PageTransform {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat img_object = Highgui.imread("samples/answerModel.jpg");
        Mat img_scene = Highgui.imread("samples/scan2.jpg");
        Mat img_out = img_scene.clone();

        if(img_scene.channels() == 3) {
            Imgproc.cvtColor(img_scene,img_scene,Imgproc.COLOR_BGR2GRAY);
        }
        Imgproc.threshold(img_scene,img_scene,50,255,Imgproc.THRESH_BINARY|Imgproc.THRESH_OTSU);
        new ImageViewer(img_scene,"阈值图").imshow();

        //使用sift检测特征点
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);
        MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
        MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        detector.detect(img_object,keypoints_object);
        detector.detect(img_scene,keypoints_scene);

        //计算特征描述子
        Mat descriptors_object = new Mat();
        Mat descriptors_scene = new Mat();
        extractor.compute(img_object,keypoints_object,descriptors_object);
        extractor.compute(img_scene,keypoints_scene,descriptors_scene);

        //使用Flann匹配算法进行匹配
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors_object,descriptors_scene,matches);

        double max_dist = 0,min_dist = 100;
        //计算特征点之间的最小和最大欧氏距离
        for(int i = 0; i<descriptors_object.rows(); i++) {
            double dist = matches.toList().get(i).distance;
            if(dist< min_dist)
                min_dist = dist;
            if(dist > max_dist)
                max_dist = dist;
        }
        MatOfDMatch good_matches = new MatOfDMatch();
        good_matches = matches;

        //绘制筛选后的特征点
        Mat img_matches = new Mat();
        //Features2d.drawMatches(img_object,keypoints_object,img_scene,keypoints_scene,good_matches,img_matches);
        //new ImageViewer(img_matches,"初始特征匹配图").imshow();

        //初始化匹配点
        List<Point> obj = new ArrayList<>();
        List<Point>  scene = new ArrayList<>();
        for(int i = 0; i<good_matches.toList().size(); i++) {
            obj.add(keypoints_object.toList().get(good_matches.toList().get(i).queryIdx).pt);
            scene.add(keypoints_scene.toList().get(good_matches.toList().get(i).trainIdx).pt);

        }

        //使用ransac算法重新筛选特征匹配点
        int ptCount = good_matches.toList().size();
        MatOfPoint2f mop_obj = new MatOfPoint2f();
        MatOfPoint2f mop_scene = new MatOfPoint2f();
        mop_obj.fromList(obj);
        mop_scene.fromList(scene);
        Mat mask = new Mat();

        Calib3d.findFundamentalMat(mop_obj, mop_scene,Calib3d.RANSAC,3,0.99,mask);

        obj.clear();
        scene.clear();

        for(int m = 0; m<ptCount; m++) {
            if(mask.get(m,0)[0] !=0) {
                obj.add(keypoints_object.toList().get(good_matches.toList().get(m).queryIdx).pt);
                scene.add(keypoints_scene.toList().get(good_matches.toList().get(m).trainIdx).pt);
            }
        }

        mop_obj.fromList(obj);
        mop_scene.fromList(scene);
        Mat H = Calib3d.findHomography(mop_obj,mop_scene,Calib3d.RANSAC,3);

        //得到模板图中的四个顶点
        List<Point> obj_corners = new ArrayList<>(4);
        List<Point> scene_corners = new ArrayList<>(4);
        obj_corners.add(new Point(0,0));
        obj_corners.add(new Point(img_object.cols(),0));
        obj_corners.add(new Point(img_object.cols(),img_object.rows()));
        obj_corners.add(new Point(0,img_object.rows()));

        Mat transform_obj = new Mat(new Size(1,4),CvType.CV_32FC2);
        System.out.println("通道数："+transform_obj.channels());
        for(int i = 0; i<transform_obj.rows(); i++) {
            transform_obj.put(i,0,obj_corners.get(i).x,obj_corners.get(i).y);
        }
        System.out.println(transform_obj.dump());
        Mat transform_scene = new Mat(new Size(1,4),CvType.CV_32FC2);
        System.out.println(H.dump());
       Core.perspectiveTransform(transform_obj,transform_scene,H);
        System.out.println(transform_scene.dump());

        for(int i = 0; i< transform_scene.height(); i++) {
            double[] data = transform_scene.get(i, 0);
            System.out.println(Arrays.toString(data));
            Core.circle(img_out,new Point(data[0],data[1]),3,new Scalar(0,0,255),2);
            scene_corners.add(new Point(data[0],data[1]));
        }
        System.out.println(img_out.size());
        new ImageViewer(img_out,"顶点图").imshow();

        //进行透视变换
        Mat result = new Mat(img_object.size(),CvType.CV_8UC1);
        result = warpTransform(img_scene, result, scene_corners);
        new ImageViewer(result,"result").imshow();

    }

    //进行透视变换
    public static Mat warpTransform(Mat src,Mat dst,List<Point> beforePoints) {
        List<Point> afterPoints = new ArrayList<Point>();
        afterPoints.add(new Point(0,0));
        afterPoints.add(new Point(dst.cols(),0));
        afterPoints.add(new Point(dst.cols(),dst.rows()));
        afterPoints.add(new Point(0,dst.rows()));

        Point[] points1 =  beforePoints.toArray(new Point[4]);
        Point[] points2 = afterPoints.toArray(new Point[1]);
//        System.out.println("points1长度："+points1.length);

//        System.out.println("points2长度："+points2.length);
//        System.out.println(points1[0]);
        Mat transform = Imgproc.getPerspectiveTransform(new MatOfPoint2f(points1),(Mat)new MatOfPoint2f(points2));
        Imgproc.warpPerspective(src, dst, transform, dst.size());
        return dst;
    }
}
