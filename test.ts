import {Collection} from "mongodb";
import {Helpers} from "./helpers";
import {MaxRequests} from "./max-requests";

const axios = require('axios');

export class WaitrExtractor {
  private static searchParamObj = [
    ['min-radius', '0'],
    ['max-radius', '20'],
    ['search_type', 'coordinates'],
  ];

  private static detailedParamObj = [
    ['stop_at_recursion_level', '3']
  ];

  private static baseDetailedUrl = 'https://api.waitrapp.com/restaurants/';
  private static baseSearchUrl = "https://api.waitrapp.com/restaurant-digests?";
  private maxRequest: MaxRequests;
  private keys = new Set();
  private authKey = "";

  constructor(private location: { lon: number, lat: number }, private bizCol: Collection, maxReqPerHour) {
    this.maxRequest = new MaxRequests(maxReqPerHour);

    this.addkeys('id').then(() => {
      this.getAuthKey().then((authKey) => {
        this.authKey = authKey;
        this.getAllRestaurants().then((searchNum) => {
          console.log('GrubHub: added ', searchNum, ' restaurants from search');
          this.getAllDetailedRestaurants().then((rNum) => {
            console.log('GrubHub: added ', searchNum, ' detailed restaurants')
          })
        })
      })
    })
  }

  private static createUrl(baseUrl: string, paramObj: string[][], coords: { lon: number, lat: number }) {

    return baseUrl + WaitrExtractor.createQueryParamsObj(paramObj) + WaitrExtractor.createLocationQueryPoint(coords)
  }

  private static createLocationQueryPoint(coords: { lon: number, lat: number }) {
    return `&longitude=${coords.lon.toString()}&latitude=${coords.lat.toString()}`
  }

  private static createQueryParamsObj(paramObj: string[][]) {
    return new URLSearchParams(paramObj)
  }

  public async getAllDetailedRestaurants() {
    let totalR = 0;
    const headers = this.getHeaderConfig();
    const restaurants = await this.bizCol.find({'hasDetails': false}).toArray();

    for (const restaurant of restaurants) {
      const baseRestaurantUrl = WaitrExtractor.baseDetailedUrl + restaurant['id'] + '/menus?';
      const url = baseRestaurantUrl + WaitrExtractor.createQueryParamsObj(WaitrExtractor.detailedParamObj);

      restaurant['details'] = (await this.getPageData(url, headers)).data;
      restaurant.hasDetails = true;
      this.bizCol.replaceOne({_id: restaurant['_id']}, restaurant);
      totalR++;
      console.log('Waitir Detailed Processed Restuarant #', totalR);
    }
    return totalR;
  }

  public async getAllRestaurants() {
    let totalSearchSize = 0;
    const headers = this.getHeaderConfig();
    const baseUrl = WaitrExtractor.createUrl(
      WaitrExtractor.baseSearchUrl, WaitrExtractor.searchParamObj, this.location
    );

    const result = (await this.getPageData(baseUrl, headers));
    const restaurants = result.data.filter((r) => {
      return !this.keys.has(r['id'])
    });
    for (const r of restaurants) {
      r['hasDetails'] = false;
    }

    if (restaurants.length > 0) {
      const writeResult = await this.bizCol.insertMany(restaurants);
      totalSearchSize += writeResult.insertedCount;
    }

    return totalSearchSize;
  }

  private async addkeys(prop: string) {
    return this.bizCol.find({}).forEach((restaurant) => {
      this.keys.add(restaurant[prop])
    }).then(() => {
      return;
    })
  }

  private async getAuthKey() {
    const url = "https://waitrapp.com/client-token?url=https:%2F%2Fgateway.waitrapp.com";
    const headers = {
      'Accept': 'application/json, text/plain, */*',
      'Accept-Language': 'en-US,en;q=0.9',
      'Cookie': 'ab.storage.deviceId.0aba6d69-9d58-4b3f-824f-f47d1b136e98=%7B%22g%22%3A%222124a58c-0f72-8270-4425-8cd4e02b6d56%22%2C%22c%22%3A1564844185780%2C%22l%22%3A1564844185780%7D; waitr.client_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6ImQ4MDE3ODJkNzNjNmE1NjQyNWExYjQ3YWEyMDk2NjAxODZjMWQyM2Y5MzM2MjY5NTE0ZTNjY2U0ZmQxN2Q4MGRhMDc1MzI2NzNhNTAxNzEwIn0.eyJhdWQiOiJ3YUFLQ25SMkVoNG11YUswN2xQZjV4a1ZXMVdSSHdqbGg3dHdQNXZKZzVObCIsImp0aSI6ImQ4MDE3ODJkNzNjNmE1NjQyNWExYjQ3YWEyMDk2NjAxODZjMWQyM2Y5MzM2MjY5NTE0ZTNjY2U0ZmQxN2Q4MGRhMDc1MzI2NzNhNTAxNzEwIiwiaWF0IjoxNTY0ODQ0MTg3LCJuYmYiOjE1NjQ4NDQxODcsImV4cCI6MTU2NDg0Nzc4Nywic3ViIjoiIiwic2NvcGVzIjpbIndhaXRyX3VzZXIiXX0.HRDqimlCYWlpj7Lm9R4t71ze18k8EwcDdHe0-Ql8JrnSx3E7NLCjAK61EOpkv-5sSVezZZjN3JO4-6NmZfMtgRYht55ismdo3NFJyWqH5XYzz4ZkzA8RW4YkItEpDIizDv_pxz0hE8xHmMCux9qFJDZBpBjtQgzWW26MDY2ixSA; intercom-id-p6vjxrta=a296c751-ee2c-4ae2-8540-b8099a77d84e; mp_mixpanel__c=5; mp_3eb666e44810de4be7201121dc9f3ef0_mixpanel=%7B%22distinct_id%22%3A%20%2216c57fb579e143-0c76b96e3a9fea-c343162-1de935-16c57fb579f365%22%2C%22%24search_engine%22%3A%20%22google%22%2C%22%24initial_referrer%22%3A%20%22https%3A%2F%2Fwww.google.com%2F%22%2C%22%24initial_referring_domain%22%3A%20%22www.google.com%22%7D; ab.storage.sessionId.0aba6d69-9d58-4b3f-824f-f47d1b136e98=%7B%22g%22%3A%22825e6ca8-e7af-472a-f111-f329163af001%22%2C%22e%22%3A1564846009553%2C%22c%22%3A1564844185776%2C%22l%22%3A1564844209553%7D',
      'Host': 'waitrapp.com',
      'Origin': 'https://www.grubhub.com',
      'Referer': 'https://waitrapp.com/cities/al/birmingham/food-delivery',
    };

    const authData = (await axios.get(url, headers)).data;
    return "Bearer " + Helpers.get_prop('token.access_token', authData);
  }

  private async getPageData(url: string, config = {}) {
    const rdy = await this.maxRequest.waitTillReady();
    return await axios.get(url, config);
  }

  private getHeaderConfig() {
    return {
      headers: {
        Authorization: this.authKey,
        Accept: 'application/json, text/plain, */*',
        Origin: 'https://waitrapp.com',
        Referer: 'https://waitrapp.com/cities/al/birmingham/food-delivery'
      }
    }
  }
}
