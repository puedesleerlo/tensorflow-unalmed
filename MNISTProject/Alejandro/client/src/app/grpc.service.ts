import { Injectable } from '@angular/core';

@Injectable()
export class GrpcService {
  grpc: any
  constructor() {
    this.grpc = require('grpc');
  }

}
